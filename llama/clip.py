__all__ = [
    '_clip_process_eval_image_embed',
    '_clip_uhd_num_image_embeds_col',
    'clip_init_context',
    'clip_free_context',
]

# from typing import Iterator

from huggingface_hub import hf_hub_download

from .llama_cpp import lib, ffi, lock, llama_context_p, clip_ctx_p, llava_image_embed_p, void_p, float_p
from .options import ModelOptions
# from .llama_cpp import lib, ffi, lock, llama_context_p, clip_ctx_p, llava_image_embed_p, void_p, llama_model_p, llama_token, llama_batch, llama_seq_id
# from .options import ModelOptions, CompletionsOptions
# from .context import context_init, context_free
# from .sampler import sampler_init, grammar_sampler_init, sampler_free
# from .formatter import get_tokenizer
# from .llava import _llava_image_embed_make_with_filename, _llava_image_embed_free
# from .util import _llama_decode, _decode_tokens, _common_sampler_sample, _common_sampler_accept, _common_token_to_piece, _common_batch_clear, _common_batch_add


def _clip_process_eval_image_embed(context: llama_context_p,
                                   clip_context: clip_ctx_p,
                                   embeds: llava_image_embed_p,
                                   n_past: int,
                                   idx: int) -> int:
    n_past_p: int_p = ffi.new('int[]', [n_past])

    with lock:
        n_batch: int = lib.llama_n_batch(context)
        image_embed: void_p = lib.malloc(lib.clip_embd_nbytes(clip_context))
        image_embed: float_p = ffi.cast('float*', image_embed)

        lib.memcpy(
            image_embed,
            embeds.embed + idx * lib.clip_n_patches(clip_context) * lib.clip_n_mmproj_embd(clip_context),
            lib.clip_embd_nbytes(clip_context),
        )

        slice_embed: void_p = lib.malloc(ffi.sizeof('struct llava_image_embed'))
        slice_embed: llava_image_embed_p = ffi.cast('struct llava_image_embed*', slice_embed)
        slice_embed.embed = image_embed
        slice_embed.n_image_pos = lib.clip_n_patches(clip_context)

        lib.llava_eval_image_embed(context, slice_embed, n_batch, n_past_p)
        lib.llava_image_embed_free(slice_embed)

    n_past = n_past_p[0]
    ffi.release(n_past_p)
    return n_past


def _clip_uhd_num_image_embeds_col(ctx_clip: clip_ctx_p) -> int:
    with lock:
        return lib.clip_uhd_num_image_embeds_col(ctx_clip)


def clip_init_context(model_options: ModelOptions) -> clip_ctx_p:
    assert model_options.ctx_size >= 2048
    assert model_options.mmproj_hf_file
    mmproj_path: str | bytes = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.mmproj_hf_file)
    # print(f'{mmproj_path=}')

    with lock:
        # clip_model_load(path, verbosity)
        clip_context: clip_ctx_p = lib.clip_model_load(mmproj_path.encode(), model_options.verbose)

    return clip_context


def clip_free_context(clip_context: clip_ctx_p):
    with lock:
        lib.clip_free(clip_context)


"""
def clip_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
    assert isinstance(completions_options.prompt, str)
    assert completions_options.messages is None, 'messages are not currently supported'
    assert isinstance(completions_options.image, str)

    _model: llama_model_p = model._model

    if completions_options.verbose:
        # default llama.cpp logger
        lib.llama_log_set(ffi.NULL, ffi.NULL)
    else:
        lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)

    context = context_init(model, model_options)
    clip_context = clip_init_context(model_options)
    sampler = sampler_init(model, completions_options)
    print(f'{sampler=}')

    if completions_options.grammar or completions_options.json_schema:
        grammar_sampler = grammar_sampler_init(model, completions_options)
    else:
        grammar_sampler = None

    print(f'{grammar_sampler=}')

    has_minicpmv_projector: int = lib.clip_is_minicpmv(clip_context)
    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)

    begin_vision_token: str = '<image>'
    end_vision_token: str = '</image>'

    if is_qwen2vl:
        begin_vision_token = '<|vision_start|>'
        end_vision_token = '<|vision_end|>'

    print(f'{has_minicpmv_projector=}')
    print(f'{is_qwen2vl=}')

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

    # create a llama_batch, we use this object to submit token data for decoding
    n_batch: int = lib.llama_n_batch(context)
    batch: llama_batch = lib.llama_batch_init(n_batch, 0, 1)
    seq_ids: list[llama_seq_id] = [0]
    n_past: int = 0
    cur_pos_id: int = n_past # used by qwen2vl
    idx: int = 0
    prompt: str
    prompt_tokens: list[int]

    # pre-embeds prompt
    messages = [{'role': 'user', 'content': begin_vision_token}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
    prompt = prompt[:prompt.index(begin_vision_token) + len(begin_vision_token)]
    print(1, f'{prompt=}')
    prompt_tokens = tokenizer.encode(prompt) # type: ignore

    if is_qwen2vl:
        # n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        prev_n_past = n_past
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        cur_pos_id += n_past - prev_n_past
    else:
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

    # embeds
    if is_qwen2vl:
        s, n_past, cur_pos_id = _qwen2vl_eval_image_embed(context, clip_context, embeds, n_batch, n_past, cur_pos_id)
        assert s
    else:
        n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

    idx += 1

    # evaluate the post-embeds prompt
    if is_qwen2vl:
        # n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        # prev_n_past = n_past
        # n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        # cur_pos_id += n_past - prev_n_past
        pass
    else:
        prompt = end_vision_token
        print(2, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore
        print(2, f'{prompt_tokens=}')

        prev_n_past = n_past
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
        cur_pos_id += n_past - prev_n_past

    print('!!! [2.0]:', n_past, cur_pos_id)

    # print(f'{has_minicpmv_projector=}')

    if has_minicpmv_projector >= 2:
        num_image_embeds = embeds.n_image_pos / lib.clip_n_patches(clip_context)

        if num_image_embeds > 1:
            num_image_embeds_col = _clip_uhd_num_image_embeds_col(clip_context)

            prompt = '<slice>'
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
            i = 0

            while i < (num_image_embeds - 1) / num_image_embeds_col:
                j = 0

                while j < num_image_embeds_col:
                    prompt = '<image>'
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    n_past = _clip_process_eval_image_embed(context, clip_context, embeds, n_past, idx)

                    idx += 1

                    prompt = '</image>'
                    prompt_tokens = tokenizer.encode(prompt) # type: ignore
                    n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                    j += 1

                prompt = '\n'
                prompt_tokens = tokenizer.encode(prompt) # type: ignore
                n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
                i += 1

            prompt = '</slice>'
            prompt_tokens = tokenizer.encode(prompt) # type: ignore
            n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)

        # _llava_image_embed_free(embeds)

        # user prompt after image ebeds
        messages = [{'role': 'user', 'content': f'\n{completions_options.prompt}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = prompt[prompt.index(f'\n{completions_options.prompt}'):]
        print(3, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore
        n_past = _decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past)
    elif is_qwen2vl:
        # user prompt after image ebeds
        messages = [{'role': 'user', 'content': f'{completions_options.prompt}'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = end_vision_token + prompt[prompt.index(f'{completions_options.prompt}'):]
        print(3, f'{prompt=}')
        prompt_tokens = tokenizer.encode(prompt) # type: ignore

        n_past, cur_pos_id = _qwen2vl_decode_tokens(context, batch, prompt_tokens, seq_ids, 0, n_past, cur_pos_id)
        print('!!! [2.1]:', n_past, cur_pos_id)

    # generate output
    n_cur: int = n_past
    n_ctx: int = lib.llama_n_ctx(context)
    n_decode: int = 0
    # print(f'{n_cur=} {n_ctx=} {n_decode=} {options.predict=}')

    while n_cur < n_ctx and n_decode < completions_options.predict:
        if grammar_sampler:
            new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
            _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)
        else:
            # new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)
            # new_token_id: llama_token = _llama_sampler_sample(sampler, context, -1)
            new_token_id: llama_token = lib.llama_sampler_sample(sampler, context, -1)

        n_decode += 1

        if lib.llama_token_is_eog(model, new_token_id):
            if n_decode > 1:
                break

        if not lib.llama_token_is_eog(model, new_token_id):
            # piece: str = tokenizer.decode(new_token_id)
            piece = _common_token_to_piece(context, new_token_id, True)
            yield piece

        # with lock:
        _common_batch_clear(batch)
        _common_batch_add(batch, new_token_id, n_past, seq_ids, True)
        batch.logits[batch.n_tokens] = True
        n_past += 1
        n_cur += 1

        r = _llama_decode(context, batch)

        if r < 0:
            raise Exception('llama_decode failed')
        elif r > 0:
            break

    # lib.llama_perf_sampler_print(sampler)
    # lib.llama_perf_context_print(context)

    _llava_image_embed_free(embeds)
    lib.llama_batch_free(batch)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)
"""
