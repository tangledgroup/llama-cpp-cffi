__all__ = ['qwen2vl_completions']

import os
from typing import Any, Optional, Iterator

from transformers import AutoTokenizer

from .options import ModelOptions, CompletionsOptions
from .llama_cpp import (
    lib,
    ffi,
    lock,
    llama_model_p,
    llama_context_p,
    llama_batch,
    llama_batch_p,
    llama_token,
    llama_vocab_p,
    clip_ctx_p,
    llava_image_embed_p,
)
from .context import context_init, context_free
from .clip import clip_init_context, clip_free_context
from .llava import _llava_image_embed_free, _llava_image_embed_make_with_filename
from .sampler import sampler_init, grammar_sampler_init, sampler_free, _common_sampler_sample, _common_sampler_accept
from .util import _llama_decode, _common_token_to_piece, _zero_array, messages_to_prompt_image
from .formatter import get_tokenizer

begin_vision_token: str = '<|vision_start|>'
end_vision_token: str = '<|vision_end|>'


def _eval_image_embed(ctx_llama: llama_context_p, ctx_clip: clip_ctx_p, image_embed: llava_image_embed_p, n_batch: int, n_past: int, st_pos_id: int) -> tuple[bool, int, int]:
    image_size: Any = lib.clip_get_load_image_size(ctx_clip)
    n_embd: int = lib.llama_n_embd(lib.llama_get_model(ctx_llama))
    patch_size: int = 14 * 2
    ph: int = int(image_size.height / patch_size + (image_size.height % patch_size > 0))
    pw: int = int(image_size.width / patch_size + (image_size.width % patch_size > 0))
    img_tokens: Any = image_embed.n_image_pos
    mrope_pos: Any = ffi.new('llama_pos[]', img_tokens * 4)

    for y in range(ph):
        for x in range(pw):
            i: int = y * pw + x
            mrope_pos[i + img_tokens * 0] = st_pos_id
            mrope_pos[i + img_tokens * 1] = st_pos_id + y
            mrope_pos[i + img_tokens * 2] = st_pos_id + x
            mrope_pos[i + img_tokens * 3] = 0

    st_pos_id += max(pw, ph)
    processed: int = 0
    batch_mrope_pos: Any = ffi.new('llama_pos[]', img_tokens * 4)

    for i in range(0, img_tokens, n_batch):
        n_eval: int = img_tokens - i

        if n_eval > n_batch:
            n_eval = n_batch

        _zero_array(batch_mrope_pos)
        ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 0), ffi.addressof(mrope_pos, img_tokens * 0 + processed), n_eval * ffi.sizeof('llama_pos'))
        ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 1), ffi.addressof(mrope_pos, img_tokens * 1 + processed), n_eval * ffi.sizeof('llama_pos'))
        ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 2), ffi.addressof(mrope_pos, img_tokens * 2 + processed), n_eval * ffi.sizeof('llama_pos'))
        ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 3), ffi.addressof(mrope_pos, img_tokens * 3 + processed), n_eval * ffi.sizeof('llama_pos'))

        embd: Any = image_embed.embed + i * n_embd

        batch_p: llama_batch_p = ffi.new('llama_batch[]', [(
            n_eval, # n_tokens
            ffi.NULL, # token
            embd, # embd
            batch_mrope_pos, # pos
            ffi.NULL, # n_seq_id
            ffi.NULL, # seq_id
            ffi.NULL, # logits
        )])

        batch: llama_batch = batch_p[0]

        if _llama_decode(ctx_llama, batch):
            ffi.release(batch_p)
            ffi.release(batch_mrope_pos)
            ffi.release(mrope_pos)
            return False, n_past, st_pos_id

        n_past += n_eval
        processed += n_eval
        ffi.release(batch_p)

    ffi.release(batch_mrope_pos)
    ffi.release(mrope_pos)
    return True, n_past, st_pos_id


def _eval_tokens(ctx_llama: llama_context_p, tokens: list[llama_token], n_batch: int, n_past, st_pos_id: int) -> tuple[bool, int, int]:
    _tokens = ffi.new('llama_token[]', tokens)

    for i in range(0, len(tokens), n_batch):
        n_eval = len(tokens) - i

        if n_eval > n_batch:
            n_eval = n_batch

        with lock:
            batch: llama_batch = lib.llama_batch_get_one(ffi.addressof(_tokens, i), n_eval)
            pos: Any = ffi.new('llama_pos[]', batch.n_tokens * 4)

            for j in range(batch.n_tokens * 3):
                pos[j] = st_pos_id + (j % batch.n_tokens)

            batch.pos = pos

            if lib.llama_decode(ctx_llama, batch):
                ffi.release(pos)
                return False, n_past, st_pos_id

        n_past += n_eval
        st_pos_id += n_eval
        ffi.release(pos)

    return True, n_past, st_pos_id


def qwen2vl_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    # either prompt/image or messages, but not both
    assert (
        (
            isinstance(completions_options.prompt, str) and
            isinstance(completions_options.image, str)
        ) and not completions_options.messages
    ) or (
        not (
            isinstance(completions_options.prompt, str) and
            isinstance(completions_options.image, str)
        ) and completions_options.messages
    )

    image_file: Optional[Any] = None

    # allow only single message
    if completions_options.messages:
        prompt, image_file = messages_to_prompt_image(completions_options.messages)
        completions_options.prompt = prompt
        completions_options.image = image_file.name # type: ignore

    _model: llama_model_p = model._model
    vocab: llama_vocab_p = lib.llama_model_get_vocab(_model)

    if completions_options.verbose:
        # default llama.cpp logger
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(_model, model_options)
    clip_context = clip_init_context(model_options)
    sampler = sampler_init(_model, completions_options)
    # print(f'{sampler=}')

    if completions_options.grammar or completions_options.json_schema:
        grammar_sampler = grammar_sampler_init(_model, completions_options)
    else:
        grammar_sampler = ffi.NULL
    # print(f'{grammar_sampler=}')

    minicpmv_projector: int = lib.clip_is_minicpmv(clip_context)
    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)
    assert not minicpmv_projector
    assert is_qwen2vl

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)

    # image embeddings
    embeds: llava_image_embed_p = _llava_image_embed_make_with_filename(
        clip_context,
        model_options.n_threads,
        completions_options.image.encode(),
    )

    assert embeds != ffi.NULL
    # print(f'{embeds=}')

    if image_file:
        os.unlink(image_file.name)

    n_past: int = 0
    cur_pos_id: int = 0
    max_tgt_len: int = 256 if completions_options.predict < 0 else completions_options.predict

    # eval user prompt
    messages = [{'role': 'user', 'content': begin_vision_token}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
    prompt = prompt[:prompt.index(begin_vision_token) + len(begin_vision_token)]
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=True) # type: ignore
    s, n_past, cur_pos_id = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past, cur_pos_id)

    # eval user image
    s, n_past, cur_pos_id = _eval_image_embed(context, clip_context, embeds, model_options.n_batch, n_past, cur_pos_id)

    # eval generation prompt for assitent
    messages = [{'role': 'user', 'content': f'{end_vision_token}{completions_options.prompt}'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    prompt = prompt[prompt.index(f'{end_vision_token}{completions_options.prompt}'):]
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
    s, n_past, cur_pos_id = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past, cur_pos_id)

    # generate tokens
    for i in range(max_tgt_len):
        new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
        _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)

        if lib.llama_token_is_eog(vocab, new_token_id):
            break

        piece = _common_token_to_piece(context, new_token_id, True)
        yield piece

        prompt_tokens: list[int] = tokenizer.encode(piece) # type: ignore
        s, n_past, cur_pos_id = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past, cur_pos_id)

    _llava_image_embed_free(embeds)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)
