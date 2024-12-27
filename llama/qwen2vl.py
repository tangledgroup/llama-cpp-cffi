__all__ = ['qwen2vl_completions']

from typing import Any, Iterator

from transformers import AutoTokenizer

from .options import ModelOptions, CompletionsOptions
from .llama_cpp import lib, ffi, llama_context_p, llama_batch, llama_seq_id, llama_batch_p, llava_image_embed_p, llama_model_p, clip_ctx_p, llama_token
from .context import context_init, context_free
from .clip import clip_init_context, clip_free_context
from .llava import _llava_image_embed_free, _llava_image_embed_make_with_filename
from .sampler import sampler_init, grammar_sampler_init, sampler_free
from .util import _common_batch_clear, _common_batch_add, _llama_decode, _decode_tokens, _common_sampler_sample, _common_sampler_accept, _common_token_to_piece
from .formatter import get_tokenizer

"""
def _qwen2vl_decode_tokens(context: llama_context_p, batch: llama_batch, prompt_tokens: list[int], seq_ids: list[llama_seq_id], n_begin: int, n_past: int, st_pos_id: int) -> tuple[int, int]:
    n_batch: int = lib.llama_n_batch(context)
    n_prompt_tokens: int = len(prompt_tokens)


    for i in range(n_begin, n_prompt_tokens, n_batch):
        _common_batch_clear(batch)

        j = 0

        while j < n_batch and i + j < n_prompt_tokens:
            _common_batch_add(batch, prompt_tokens[i + j], n_past, seq_ids, False)
            n_past += 1
            st_pos_id += 1
            j += 1

        if i + n_batch >= n_prompt_tokens:
            batch.logits[batch.n_tokens - 1] = True

        j = 0

        print(f'!!! {batch.n_tokens=}, {batch.n_tokens * 4=}')

        pos: Any = ffi.new('llama_pos[]', [
            int(st_pos_id + (j % batch.n_tokens))
            for j in range(batch.n_tokens * 4)
        ])

        print(f'{pos=}')
        batch.pos = pos

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

        if i + n_batch >= n_prompt_tokens:
            break

        ffi.release(pos)

    return n_past, st_pos_id
"""

def _qwen2vl_eval_image_embed(ctx_llama: llama_context_p, ctx_clip: clip_ctx_p, image_embed: llava_image_embed_p, n_batch: int, n_past: int, st_pos_id: int) -> tuple[bool, int, int]:
    print('!!! [0]:', n_past, st_pos_id)
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
    print('!!! [*]:', n_past, st_pos_id)

    for i in range(0, img_tokens, n_batch):
        n_eval: int = img_tokens - i

        if n_eval > n_batch:
            n_eval = n_batch

        # for i in range(img_tokens * 4):
        #     batch_mrope_pos[i] = 0

        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 0), ffi.addressof(mrope_pos, img_tokens * 0 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 1), ffi.addressof(mrope_pos, img_tokens * 1 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 2), ffi.addressof(mrope_pos, img_tokens * 2 + processed), n_eval * ffi.sizeof('llama_pos'))
        lib.memcpy(ffi.addressof(batch_mrope_pos, n_eval * 3), ffi.addressof(mrope_pos, img_tokens * 3 + processed), n_eval * ffi.sizeof('llama_pos'))

        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 0), ffi.addressof(mrope_pos, img_tokens * 0 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 1), ffi.addressof(mrope_pos, img_tokens * 1 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 2), ffi.addressof(mrope_pos, img_tokens * 2 + processed), n_eval * ffi.sizeof('llama_pos'))
        # ffi.memmove(ffi.addressof(batch_mrope_pos, n_eval * 3), ffi.addressof(mrope_pos, img_tokens * 3 + processed), n_eval * ffi.sizeof('llama_pos'))

        embd: Any = image_embed.embed + i * n_embd
        # embd[0] = embd[0]
        # batch_mrope_pos[0] = batch_mrope_pos[0]
        # global_weakkeydict[image_embed] = embd
        print(f'{n_eval=}')
        print(f'{embd=}')
        print(f'{batch_mrope_pos=}')

        batch_p: llama_batch_p = ffi.new('llama_batch[]', [(
            n_eval, # n_tokens
            ffi.NULL, # token
            embd, # embd
            batch_mrope_pos, # pos
            ffi.NULL, # n_seq_id
            ffi.NULL, # seq_id
            ffi.NULL, # logits
        )])

        # global_weakkeydict[batch_p] = (embd, batch_mrope_pos)

        batch: llama_batch = batch_p[0]
        # global_weakkeydict[batch_p] = batch

        if _llama_decode(ctx_llama, batch):
            return False, n_past, st_pos_id

        n_past += n_eval
        processed += n_eval
        # ffi.release(batch_p)
        print(f'!!! {i=}')

    print('!!! [1]:', n_past, st_pos_id)
    return True, n_past, st_pos_id


def _eval_tokens(ctx_llama: llama_context_p, tokens: list[llama_token], n_batch: int, n_past, st_pos_id: int) -> tuple[bool, int, int]:
    N: int = len(tokens)
    _tokens = ffi.new('llama_token[]', tokens)

    for i in range(0, N, n_batch):
        n_eval = N - i

        if n_eval > n_batch:
            n_eval = n_batch

        batch: llama_batch = lib.llama_batch_get_one(ffi.addressof(_tokens, i), n_eval)
        pos: Any = ffi.new('llama_pos[]', batch.n_tokens * 4)

        for j in range(batch.n_tokens * 3):
            pos[j] = st_pos_id + (j % batch.n_tokens)

        batch.pos = pos

        if lib.llama_decode(ctx_llama, batch):
            return False, n_past, st_pos_id

        ffi.release(pos)
        n_past += n_eval
        st_pos_id += n_eval

    return True, n_past, st_pos_id


def qwen2vl_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    assert isinstance(completions_options.prompt, str)
    assert completions_options.messages is None, 'messages are not currently supported'
    assert isinstance(completions_options.image, str)

    _model: llama_model_p = model._model

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

    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)
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
        model_options.threads,
        completions_options.image.encode(),
    )

    assert embeds != ffi.NULL
    # print(f'{embeds=}')

    n_past: int = 0
    cur_pos_id: int = 0
    max_tgt_len: int = 256 if completions_options.predict < 0 else completions_options.predict

    prompt = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|>'
    prompt_tokens: list[int] = tokenizer.encode(prompt) # type: ignore
    s, n_past, cur_pos_id = _eval_tokens(context, prompt_tokens, model_options.batch_size, n_past, cur_pos_id)

    if embeds is not ffi.NULL:
        s, n_past, cur_pos_id = _qwen2vl_eval_image_embed(context, clip_context, embeds, model_options.batch_size, n_past, cur_pos_id)

    prompt = '<|vision_end|>' + completions_options.prompt + '<|im_end|>\n<|im_start|>assistant\n'
    prompt_tokens: list[int] = tokenizer.encode(prompt) # type: ignore
    s, n_past, cur_pos_id = _eval_tokens(context, prompt_tokens, model_options.batch_size, n_past, cur_pos_id)
    print('!!! [2]:', n_past, cur_pos_id)

    # yield '[END]'

    for i in range(max_tgt_len):
        new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
        _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)

        if lib.llama_token_is_eog(_model, id):
            break

        piece = _common_token_to_piece(context, new_token_id, True)
        yield piece

    _llava_image_embed_free(embeds)
    # lib.llama_batch_free(batch)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)
