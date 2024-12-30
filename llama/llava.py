__all__ = [
    '_llava_image_embed_make_with_filename',
    '_llava_image_embed_free',
    'llava_completions',
]

from typing import Any, Iterator

from transformers import AutoConfig, AutoTokenizer

from .options import ModelOptions, CompletionsOptions
from .llama_cpp import lib, ffi, lock, llama_context_p, llama_batch, llama_batch_p, llava_image_embed_p, llama_model_p, clip_ctx_p, llama_token, int_p
from .context import context_init, context_free
from .clip import clip_init_context, clip_free_context
from .sampler import sampler_init, grammar_sampler_init, sampler_free, _common_sampler_sample, _common_sampler_accept
from .util import _llama_decode, _common_token_to_piece, _zero_array
from .formatter import get_config, get_tokenizer


vision_token: str = '<image>'


def _llava_image_embed_make_with_filename(ctx_clip: clip_ctx_p, n_threads: int, image_path: bytes) -> llava_image_embed_p:
    embed: llava_image_embed_p

    with lock:
        embed = lib.llava_image_embed_make_with_filename(ctx_clip, n_threads, image_path)

    return embed


def _llava_image_embed_free(embed: llava_image_embed_p):
    with lock:
        lib.llava_image_embed_free(embed)


def _eval_tokens(ctx_llama: llama_context_p, tokens: list[llama_token], n_batch: int, n_past) -> tuple[bool, int]:
    _tokens = ffi.new('llama_token[]', tokens)

    for i in range(0, len(tokens), n_batch):
        n_eval = len(tokens) - i

        if n_eval > n_batch:
            n_eval = n_batch

        batch: llama_batch = lib.llama_batch_get_one(ffi.addressof(_tokens, i), n_eval)

        if lib.llama_decode(ctx_llama, batch):
            return False, n_past

        n_past += n_eval

    return True, n_past


def llava_completions(model: 'Model', model_options: ModelOptions, completions_options: CompletionsOptions) -> Iterator[str]:
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

    config: AutoConfig = get_config(model_options.creator_hf_repo)
    model_type: str = config.model_type # type: ignore

    # tokenizer
    tokenizer: AutoTokenizer

    if model_options.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model_options.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model_options.creator_hf_repo)

    minicpmv_projector: int = lib.clip_is_minicpmv(clip_context)
    is_qwen2vl: bool = lib.clip_is_qwen2vl(clip_context)
    assert not minicpmv_projector
    assert not is_qwen2vl

    # image embeddings
    embeds: llava_image_embed_p = _llava_image_embed_make_with_filename(
        clip_context,
        model_options.threads,
        completions_options.image.encode(),
    )

    assert embeds != ffi.NULL
    # print(f'{embeds=}')

    n_past: int = 0
    max_tgt_len: int = 256 if completions_options.predict < 0 else completions_options.predict

    # eval user prompt
    if model_type == 'moondream1':
        prompt = ''
    elif model_type == 'llava-qwen2':
        messages = [{'role': 'user', 'content': vision_token}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
        prompt = prompt[:prompt.index(vision_token)]
    elif model_type == 'bunny-phi3':
        # messages = [
        #     {'role': 'system', 'content': "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."},
        #     {'role': 'user', 'content': vision_token},
        # ]
        # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
        # prompt = prompt[:prompt.index(vision_token)]

        # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        # prompt += " USER: "

        prompt = "USER: "
        # prompt = f"USER: {completions_options.prompt}"
    else:
        messages = [{'role': 'user', 'content': completions_options.prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) # type: ignore
        prompt = prompt[:prompt.index(completions_options.prompt) + len(completions_options.prompt)]

    print(prompt)

    if model_type == 'bunny-phi3':
        # do not generate double BOS
        prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
        prompt_tokens += [-200]
    else:
        prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=True) # type: ignore

    print(f'{prompt_tokens=}')
    s, n_past = _eval_tokens(context, prompt_tokens, model_options.batch_size, n_past)

    # eval user image
    n_past_p: int_p = ffi.new('int[]', [n_past])
    lib.llava_eval_image_embed(context, embeds, model_options.batch_size, n_past_p)
    n_past = n_past_p[0]
    ffi.release(n_past_p)

    # eval generation prompt for assitent
    if model_type == 'moondream1':
        prompt = f'\n\nQuestion: {completions_options.prompt}\n\nAnswer:'
    elif model_type == 'llava-qwen2':
        messages = [{'role': 'user', 'content': completions_options.prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = prompt[prompt.index(completions_options.prompt):]
    elif model_type == 'bunny-phi3':
        # messages = [{'role': 'user', 'content': vision_token + completions_options.prompt}]
        # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        # prompt = prompt[prompt.index(vision_token) + len(vision_token):]

        prompt = f'\n{completions_options.prompt}\nASSISTANT:'

        # prompt = '\nASSISTANT:'
    else:
        messages = [{'role': 'user', 'content': completions_options.prompt}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) # type: ignore
        prompt = prompt[prompt.index(completions_options.prompt) + len(completions_options.prompt):]

    print(prompt)
    prompt_tokens: list[int] = tokenizer.encode(prompt, add_special_tokens=False) # type: ignore
    print(f'{prompt_tokens=}')
    s, n_past = _eval_tokens(context, prompt_tokens, model_options.batch_size, n_past)

    # generate tokens
    for i in range(max_tgt_len):
        new_token_id: llama_token = _common_sampler_sample(grammar_sampler, sampler, context, -1, False)
        _common_sampler_accept(grammar_sampler, sampler, new_token_id, True)

        if lib.llama_token_is_eog(_model, new_token_id):
            break

        # piece = _common_token_to_piece(context, new_token_id, True)
        piece = _common_token_to_piece(context, new_token_id, False)
        yield piece

        prompt_tokens: list[int] = tokenizer.encode(piece, add_special_tokens=False) # type: ignore
        s, n_past = _eval_tokens(context, prompt_tokens, model_options.batch_size, n_past)
        # print(f'{n_past=}')

    _llava_image_embed_free(embeds)

    if grammar_sampler:
        sampler_free(grammar_sampler)

    sampler_free(sampler)
    clip_free_context(clip_context)
    context_free(context)
