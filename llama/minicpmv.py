__all__ = [
    'minicpmv_completions',
]

import os
from typing import Any, Optional, Iterator

from transformers import AutoTokenizer

from .options import ModelOptions, CompletionsOptions
from .llama_cpp import (
    lib,
    ffi,
    lock,
    int_p,
    void_p,
    float_p,
    llama_model_p,
    llama_context_p,
    llama_batch,
    llama_token,
    llama_vocab_p,
    clip_ctx_p,
    llava_image_embed_p,
)
from .context import context_init, context_free
from .clip import clip_init_context, clip_free_context
from .sampler import sampler_init, grammar_sampler_init, sampler_free, _common_sampler_sample, _common_sampler_accept
from .util import _common_token_to_piece, messages_to_prompt_image
from .formatter import get_tokenizer
from .llava import _llava_image_embed_make_with_filename, _llava_image_embed_free
from .clip import _clip_uhd_num_image_embeds_col, _clip_process_eval_image_embed


begin_image_token: str = '<image>'
end_image_token: str = '</image>'
begin_slice_token: str = '<slice>'
end_slice_token: str = '</slice>'


def _eval_tokens(ctx_llama: llama_context_p, tokens: list[llama_token], n_batch: int, n_past) -> tuple[bool, int]:
    _tokens = ffi.new('llama_token[]', tokens)

    for i in range(0, len(tokens), n_batch):
        n_eval = len(tokens) - i

        if n_eval > n_batch:
            n_eval = n_batch

        with lock:
            batch: llama_batch = lib.llama_batch_get_one(ffi.addressof(_tokens, i), n_eval)

            if lib.llama_decode(ctx_llama, batch):
                return False, n_past

        n_past += n_eval

    return True, n_past


def _process_eval_image_embed(ctx_llama: llama_context_p, ctx_clip: clip_ctx_p, embeds: llava_image_embed_p, n_batch: int, n_past: int, idx: int):
    n_past_p: int_p = ffi.new('int[]', [n_past])

    with lock:
        # n_batch: int = lib.llama_n_batch(ctx_llama)
        image_embed: void_p = lib.malloc(lib.clip_embd_nbytes(ctx_clip))
        image_embed: float_p = ffi.cast('float*', image_embed)

        lib.memcpy(
            image_embed,
            embeds.embed + idx * lib.clip_n_patches(ctx_clip) * lib.clip_n_mmproj_embd(ctx_clip),
            lib.clip_embd_nbytes(ctx_clip),
        )

        slice_embed: void_p = lib.malloc(ffi.sizeof('struct llava_image_embed'))
        slice_embed: llava_image_embed_p = ffi.cast('struct llava_image_embed*', slice_embed)
        slice_embed.embed = image_embed
        slice_embed.n_image_pos = lib.clip_n_patches(ctx_clip)

        lib.llava_eval_image_embed(ctx_llama, slice_embed, n_batch, n_past_p)
        lib.llava_image_embed_free(slice_embed)

    n_past = n_past_p[0]
    ffi.release(n_past_p)
    return n_past


def minicpmv_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
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

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)

    minicpmv_projector: int = lib.clip_is_minicpmv(clip_context)
    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)
    assert minicpmv_projector
    assert not is_qwen2vl

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
    max_tgt_len: int = 256 if completions_options.predict < 0 else completions_options.predict
    idx: int = 0

    #
    messages = [{'role': 'user', 'content': begin_image_token}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
    prompt = prompt[:prompt.index(begin_image_token) + len(begin_image_token)]
    # print(prompt)
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
    s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

    # eval user image
    n_past = _process_eval_image_embed(context, clip_context, embeds, model_options.n_batch, n_past, idx)
    idx += 1

    #
    prompt = end_image_token
    # print(prompt)
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
    s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

    #
    if minicpmv_projector > 1:
        num_image_embeds: int = int(embeds.n_image_pos / lib.clip_n_patches(clip_context))

        if num_image_embeds > 1:
            num_image_embeds_col = _clip_uhd_num_image_embeds_col(clip_context)

            prompt = '<slice>'
            # print(prompt)
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)
            i = 0

            while i < (num_image_embeds - 1) / num_image_embeds_col:
                j = 0

                while j < num_image_embeds_col:
                    prompt = '<image>'
                    # print(prompt)
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)
                    n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)
                    idx += 1

                    prompt = '</image>'
                    # print(prompt)
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

                    j += 1

                prompt = '\n'
                prompt_tokens = tokenizer.encode(prompt) # type: ignore
                s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

                i += 1

            prompt = '</slice>'
            # print(prompt)
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

    #
    messages = [{'role': 'user', 'content': completions_options.prompt}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
    prompt = prompt[prompt.index(completions_options.prompt):]
    # print(prompt)
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
    s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)

    # generate tokens
    for i in range(max_tgt_len):
        new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
        _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)

        if lib.llama_token_is_eog(vocab, new_token_id):
            break

        piece = _common_token_to_piece(context, new_token_id, True)
        yield piece

        prompt_tokens: list[int] = tokenizer.encode(piece, add_special_tokens=False) # type: ignore
        s, n_past = _eval_tokens(context, prompt_tokens, model_options.n_batch, n_past)
        # print(f'{n_past=}')

    _llava_image_embed_free(embeds)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)
