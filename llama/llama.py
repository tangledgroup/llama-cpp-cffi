__all__ = [
    # high-level API
    # 'completions',

    # low-level API
    'lib',
    'ffi',
    'backend_init',
    'backend_free',
    'numa_init',
    'model_init',
    'model_free',
    'context_init',
    'context_free',
    'sampler_init',
    'sampler_free',
    'clip_init_context',
    'clip_free_context',
    'mllama_init_context',
    'mllama_free_context',
    'text_completions',
    'clip_completions',
    'mllama_completions',
]

import os
# import ctypes
# from queue import Queue
# from copy import deepcopy
# from typing import Any, Optional, Iterator, Callable, NewType
from typing import Optional, Iterator, NewType
from threading import Lock
# from functools import partial

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# from .model import Model
from .options import Options
from .util import is_cuda_available, is_vulkan_available
from .formatter import get_tokenizer, format_messages, VLM_TEMPLATE


os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'true')
os.environ['GGML_VK_DISABLE_COOPMAT'] = os.getenv('GGML_VK_DISABLE_COOPMAT', '1')

LLAMA_CPP_BACKEND = os.getenv('LLAMA_CPP_BACKEND', None)

try:
    if LLAMA_CPP_BACKEND:
        if LLAMA_CPP_BACKEND in ('cuda', 'CUDA'):
            from ._llama_cpp_cuda_12_6_3 import lib, ffi
        elif LLAMA_CPP_BACKEND in ('vulkan', 'VULKAN'):
            from ._llama_cpp_vulkan_1_x import lib, ffi
        elif LLAMA_CPP_BACKEND in ('cpu', 'CPU'):
            from ._llama_cpp_cpu import lib, ffi
        else:
            raise ValueError(f'{LLAMA_CPP_BACKEND = }')
    else:
        if is_cuda_available():
            from ._llama_cpp_cuda_12_6_3 import lib, ffi
        elif is_vulkan_available():
            from ._llama_cpp_vulkan_1_x import lib, ffi
        else:
            from ._llama_cpp_cpu import lib, ffi
except ImportError:
    from ._llama_cpp_cpu import lib, ffi


#
# low-level API
#
void_p = NewType('void*', ffi.typeof('void*'))
char_p = NewType('char*', ffi.typeof('char*'))
int_p = NewType('int*', ffi.typeof('int*'))
float_p = NewType('float*', ffi.typeof('float*'))
ggml_numa_strategy  = NewType('ggml_numa_strategy', ffi.typeof('enum ggml_numa_strategy'))
llama_model_params  = NewType('llama_model_params', ffi.typeof('struct llama_model_params'))
llama_model  = NewType('llama_model', ffi.typeof('struct llama_model'))
llama_model_p  = NewType('llama_model*', ffi.typeof('struct llama_model*'))
llama_context  = NewType('llama_context', ffi.typeof('struct llama_context'))
llama_context_p  = NewType('llama_context*', ffi.typeof('struct llama_context*'))
llama_context_params  = NewType('llama_context_params', ffi.typeof('struct llama_context_params'))
llama_sampler  = NewType('llama_sampler', ffi.typeof('struct llama_sampler'))
llama_sampler_p  = NewType('llama_sampler*', ffi.typeof('struct llama_sampler*'))
llama_sampler_chain_params  = NewType('llama_sampler_chain_params', ffi.typeof('struct llama_sampler_chain_params'))
llama_batch  = NewType('llama_batch', ffi.typeof('struct llama_batch'))
clip_ctx  = NewType('clip_ctx', ffi.typeof('struct clip_ctx'))
clip_ctx_p  = NewType('clip_ctx*', ffi.typeof('struct clip_ctx*'))
mllama_ctx  = NewType('mllama_ctx', ffi.typeof('struct mllama_ctx'))
mllama_ctx_p  = NewType('mllama_ctx*', ffi.typeof('struct mllama_ctx*'))
llava_image_embed  = NewType('llava_image_embed', ffi.typeof('struct llava_image_embed'))
llava_image_embed_p  = NewType('llava_image_embed*', ffi.typeof('struct llava_image_embed*'))
mllama_image  = NewType('mllama_image', ffi.typeof('struct mllama_image'))
mllama_image_p  = NewType('mllama_image*', ffi.typeof('struct mllama_image*'))

lock = Lock()


def backend_init():
    lib.llama_backend_init()


def backend_free():
    lib.llama_backend_free()


def numa_init(numa: ggml_numa_strategy):
    lib.llama_numa_init(numa)


def model_init(options: Options) -> llama_model_p:
    model_path = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.hf_file)

    model_params: llama_model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = options.gpu_layers

    model: llama_model_p = lib.llama_load_model_from_file(model_path.encode(), model_params)
    return model


def model_free(model: llama_model_p):
    lib.llama_free_model(model)


def context_init(model: llama_model_p, options: Options) -> llama_context_p:
    ctx_params: llama_context_params = lib.llama_context_default_params()
    ctx_params.n_ctx = options.ctx_size
    ctx_params.n_batch = options.batch_size

    context = lib.llama_new_context_with_model(model, ctx_params)
    return context


def context_free(context: llama_context_p):
    lib.llama_free(context)


def sampler_init(model: llama_model_p, options: Options) -> llama_sampler_p:
    sampler_params: llama_sampler_chain_params = lib.llama_sampler_chain_default_params()
    sampler: llama_sampler_p = lib.llama_sampler_chain_init(sampler_params)

    # dry
    seq_breakers: char_p = ffi.new('char*[]', options.dry_sequence_breaker)
    num_breakers: int = len(options.dry_sequence_breaker)

    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dry(
        model,
        options.dry_multiplier,
        options.dry_base,
        options.dry_allowed_length,
        options.dry_penalty_last_n,
        seq_breakers,
        num_breakers,
    ))

    # common
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp(options.temp))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(options.top_k))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_p(options.top_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_min_p(options.min_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(options.seed))

    # penalties
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_penalties(
        lib.llama_n_vocab(model),
        lib.llama_token_eos(model),
        lib.llama_token_nl(model),
        options.repeat_last_n,      # last n tokens to penalize (0 = disable penalty, -1 = context size)
        options.repeat_penalty,     # 1.0 = disabled
        options.frequency_penalty,  # 0.0 = disabled
        options.presence_penalty,   # 0.0 = disabled
        options.penalize_nl,        # consider newlines as a repeatable token
        options.ignore_eos,         # ignore the end-of-sequence token
    ))

    """
    # grammar
    if options.grammar:
        grammar_str: char_p = ffi.new('char[]', options.grammar.encode())
        grammar_root: char_p = ffi.new('char[]', b'root')

        grmr = lib.llama_sampler_init_grammar(
            model,
            grammar_str,
            grammar_root,
        )

    if options.json_schema:
        assert options.json_schema == '{}'

        grammar = r'''
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
    "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
    )? "}" ws

array  ::=
    "[" ws (
            value
    ("," ws value)*
    )? "]" ws

string ::=
    "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt] | "u" [0-9a-fA-F]{4}) # escapes
    )* "\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws

# Optional space: by convention, applied in this grammar after literal chars when allowed
ws ::= | " " | "\n" [ \t]{0,20}
        '''

        grammar_str: char_p = ffi.new('char[]', grammar.encode())
        grammar_root: char_p = ffi.new('char[]', b'root')

        grmr = lib.llama_sampler_init_grammar(
            model,
            grammar_str,
            grammar_root,
        )
    """

    return sampler


def grammar_sampler_init(model: llama_model_p, options: Options) -> llama_sampler_p:
    pass


def sampler_free(sampler: llama_sampler_p):
    lib.llama_sampler_free(sampler)


def clip_init_context(options: Options) -> clip_ctx_p:
    mmproj_path: str = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.mmproj_hf_file)
    clip_context: clip_ctx_p = lib.clip_model_load(mmproj_path.encode(), 1)
    return clip_context


def clip_free_context(clip_context: clip_ctx_p):
    lib.clip_free(clip_context)


def mllama_init_context(options: Options) -> mllama_ctx_p:
    mmproj_path: str = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.mmproj_hf_file)
    mllama_context: mllama_ctx_p = lib.mllama_model_load(mmproj_path.encode(), 1)
    return mllama_context


def mllama_free_context(mllama_context: mllama_ctx_p):
    lib.mllama_free(mllama_context)


#
# util
#
def check_context_size(context: llama_context_p, batch: llama_batch) -> int:
    n_ctx: int = lib.llama_n_ctx(context)
    n_ctx_used: int = lib.llama_get_kv_cache_used_cells(context)

    if n_ctx_used + batch.n_tokens > n_ctx:
        return 1

    return 0


def batch_get_one_and_decode(context: llama_context_p,
                             prompt: str,
                             n_batch: int,
                             n_past: Optional[int_p],
                             tokenizer: AutoTokenizer) -> Optional[llama_batch]:
    prompt_tokens: list[int] = tokenizer.encode(prompt)
    n_prompt_tokens: int = len(prompt_tokens)

    for i in range(0, n_prompt_tokens, n_batch):
        sub_prompt_tokens = prompt_tokens[i:i+n_batch]
        n_sub_prompt_tokens = len(sub_prompt_tokens)
        _prompt_tokens = ffi.new('llama_token[]', sub_prompt_tokens)
        batch: llama_batch = lib.llama_batch_get_one(_prompt_tokens, n_prompt_tokens)

        if lib.llama_decode(context, batch):
            ffi.release(_prompt_tokens)
            return None

        if n_past is not None:
            n_past[0] += n_sub_prompt_tokens

        ffi.release(_prompt_tokens)

    return batch


#
# llm
#
def text_completions(model: llama_model_p,
                     context: llama_context_p,
                     sampler: llama_sampler_p,
                     # grmr_sampler: llama_sampler_p,
                     options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str) or isinstance(options.messages, list)

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # format messages if present
    if options.messages:
        options.prompt = format_messages(tokenizer, options.messages, options)

    # first batch
    prompt_tokens: list[int] = tokenizer.encode(options.prompt)
    n_prompt_tokens: int = len(prompt_tokens)
    _prompt_tokens = ffi.new('llama_token[]', prompt_tokens)
    batch: llama_batch = lib.llama_batch_get_one(_prompt_tokens, n_prompt_tokens)

    total_n_prompt_tokens: int = n_prompt_tokens
    new_token_id: int
    _new_prompt_tokens = None

    while True:
        if options.predict == -1:
            pass
        elif options.predict == -2:
            if check_context_size(context, batch):
                break
        elif total_n_prompt_tokens >= options.predict:
            break

        with lock:
            if lib.llama_decode(context, batch):
                break

        new_token_id = lib.llama_sampler_sample(sampler, context, -1)

        if lib.llama_token_is_eog(model, new_token_id):
            break

        piece: str = tokenizer.decode([new_token_id])
        yield piece

        # next batch
        if _new_prompt_tokens is not None:
            ffi.release(_new_prompt_tokens)

        _new_prompt_tokens = ffi.new('llama_token[]', [new_token_id])
        batch = lib.llama_batch_get_one(_new_prompt_tokens, 1)
        total_n_prompt_tokens += 1

    if _new_prompt_tokens is not None:
        ffi.release(_new_prompt_tokens)

#
# clip
#
def clip_process_eval_image_embed(context: llama_context_p,
                                  clip_context: clip_ctx_p,
                                  embeds: llava_image_embed_p,
                                  n_batch: int,
                                  n_past: int_p,
                                  idx: int):
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
    lib.llava_eval_image_embed(context, slice_embed, n_batch, n_past)
    lib.llava_image_embed_free(slice_embed)


def clip_completions(model: llama_model_p,
                     context: llama_context_p,
                     sampler: llama_sampler_p,
                     # grmr_sampler: llama_sampler_p,
                     clip_context: clip_ctx_p,
                     options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str) or isinstance(options.messages, list)
    assert isinstance(options.image, str) or isinstance(options.messages, list)

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # format messages if present
    if options.messages:
        # options.chat_template = VLM_TEMPLATE
        options.prompt = format_messages(tokenizer, options.messages, options)
        print('options.prompt[0]', options.prompt)

    # llava - process image
    with lock:
        n_past: int_p = ffi.new('int*', 0)
        embeds: llava_image_embed_p = lib.llava_image_embed_make_with_filename(clip_context, options.threads, options.image.encode())
        idx: int = 0

        messages = [{'role': 'user', 'content': '<image>'}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # print('prompt [0]:', prompt)
        prompt = prompt[:prompt.index('<image>') + len('<image>')]
        # print('prompt [1]:', prompt)

        batch = batch_get_one_and_decode(context, prompt, options.batch_size, n_past, tokenizer)
        clip_process_eval_image_embed(context, clip_context, embeds, options.batch_size, n_past, idx)
        idx += 1
        batch = batch_get_one_and_decode(context, '</image>', options.batch_size, n_past, tokenizer)

        has_minicpmv_projector = lib.clip_is_minicpmv(clip_context)
        # print(f'{has_minicpmv_projector=}')

        if has_minicpmv_projector >= 2:
            num_image_embeds = embeds.n_image_pos / lib.clip_n_patches(clip_context)

            if num_image_embeds > 1:
                num_image_embeds_col = lib.clip_uhd_num_image_embeds_col(clip_context)
                batch = batch_get_one_and_decode(context, '<slice>', options.batch_size, n_past, tokenizer)
                i = 0

                while i < (num_image_embeds - 1) / num_image_embeds_col:
                    j = 0

                    while j < num_image_embeds_col:
                        batch = batch_get_one_and_decode(context, '<image>', options.batch_size, n_past, tokenizer)
                        clip_process_eval_image_embed(context, clip_context, embeds, options.batch_size, n_past, idx)
                        idx += 1
                        batch = batch_get_one_and_decode(context, '</image>', options.batch_size, n_past, tokenizer)
                        j += 1

                    batch = batch_get_one_and_decode(context, '\n', options.batch_size, n_past, tokenizer)
                    i += 1

                batch = batch_get_one_and_decode(context, '</slice>', options.batch_size, n_past, tokenizer)

        lib.llava_image_embed_free(embeds)

    # first batch
    messages = [{'role': 'user', 'content': f'\n{options.prompt}'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt[prompt.index(f'\n{options.prompt}'):]
    # print('prompt [2]:', prompt)

    prompt_tokens: list[int] = tokenizer.encode(prompt)
    n_prompt_tokens: int = len(prompt_tokens)
    _prompt_tokens = ffi.new('llama_token[]', prompt_tokens)
    batch: llama_batch = lib.llama_batch_get_one(_prompt_tokens, n_prompt_tokens)

    total_n_prompt_tokens: int = n_prompt_tokens # FIXME: get from context
    new_token_id: int
    _new_prompt_tokens = None

    while True:
        if options.predict == -1:
            pass
        elif options.predict == -2:
            if check_context_size(context, batch):
                break
        elif total_n_prompt_tokens >= options.predict:
            break

        with lock:
            if lib.llama_decode(context, batch):
                break

        new_token_id = lib.llama_sampler_sample(sampler, context, -1)

        if lib.llama_token_is_eog(model, new_token_id):
            break

        token: str = tokenizer.decode(new_token_id)
        yield token

        # next batch
        if _new_prompt_tokens is not None:
            ffi.release(_new_prompt_tokens)

        _new_prompt_tokens = ffi.new('llama_token[]', [new_token_id])
        batch = lib.llama_batch_get_one(_new_prompt_tokens, 1)
        total_n_prompt_tokens += 1

    if _new_prompt_tokens is not None:
        ffi.release(_new_prompt_tokens)

    ffi.release(n_past)


#
# TODO: mllama, work in progress
#
def mllama_process_eval_image_embed(context: llama_context_p,
                                    mllama_context: mllama_ctx_p,
                                    embeds: llava_image_embed_p,
                                    n_batch: int,
                                    n_past: int_p,
                                    idx: int):
    image_embed: void_p = lib.malloc(lib.mllama_n_embd_bytes(mllama_context))
    image_embed: float_p = ffi.cast('float*', image_embed)

    lib.memcpy(
        image_embed,
        embeds.embed + idx * lib.mllama_n_patches(mllama_context) * lib.mllama_n_embd(mllama_context),
        lib.mllama_n_embd_bytes(mllama_context),
    )

    slice_embed: void_p = lib.malloc(ffi.sizeof('struct llava_image_embed'))
    slice_embed: llava_image_embed_p = ffi.cast('struct llava_image_embed*', slice_embed)
    slice_embed.embed = image_embed
    slice_embed.n_image_pos = lib.mllama_n_patches(mllama_context)
    lib.llava_eval_image_embed(context, slice_embed, n_batch, n_past)
    lib.llava_image_embed_free(slice_embed)


def mllama_completions(model: llama_model_p,
                       context: llama_context_p,
                       sampler: llama_sampler_p,
                       # grmr_sampler: llama_sampler_p,
                       mllama_context: mllama_ctx_p,
                       options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str) or isinstance(options.messages, list)
    assert isinstance(options.image, str)

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # format messages if present
    if options.messages:
        options.prompt = format_messages(tokenizer, options.messages, options)

    # llava - process image
    n_past: int_p = ffi.new('int*', 0)
    # embeds: llava_image_embed_p = lib.llava_image_embed_make_with_filename(mllama_context, options.threads, options.image.encode())
    idx: int = 0

    messages = [{'role': 'user', 'content': '<image>'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = prompt[:prompt.index('<image>') + len('<image>')]
    print('prompt [0]:', prompt)

    batch = batch_get_one_and_decode(context, prompt, options.batch_size, n_past, tokenizer)
    # mllama_process_eval_image_embed(context, mllama_context, embeds, options.batch_size, n_past, idx)
    # idx += 1
    batch = batch_get_one_and_decode(context, '</image>', options.batch_size, n_past, tokenizer)

    # lib.llava_image_embed_free(embeds)

    # first batch
    messages = [{'role': 'user', 'content': f'\n{options.prompt}'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = prompt[prompt.index(f'\n{options.prompt}'):]
    print('prompt [1]:', prompt)

    prompt_tokens: list[int] = tokenizer.encode(prompt)
    n_prompt_tokens: int = len(prompt_tokens)
    _prompt_tokens = ffi.new('llama_token[]', prompt_tokens)
    batch: llama_batch = lib.llama_batch_get_one(_prompt_tokens, n_prompt_tokens)

    total_n_prompt_tokens: int = n_prompt_tokens # FIXME: get from context
    new_token_id: int
    _new_prompt_tokens = None

    while True:
        if options.predict == -1:
            pass
        elif options.predict == -2:
            if check_context_size(context, batch):
                break
        elif total_n_prompt_tokens >= options.predict:
            break

        with lock:
            if lib.llama_decode(context, batch):
                break

        new_token_id = lib.llama_sampler_sample(sampler, context, -1)

        if lib.llama_token_is_eog(model, new_token_id):
            break

        token: str = tokenizer.decode(new_token_id)
        yield token

        # next batch
        if _new_prompt_tokens is not None:
            ffi.release(_new_prompt_tokens)

        _new_prompt_tokens = ffi.new('llama_token[]', [new_token_id])
        batch = lib.llama_batch_get_one(_new_prompt_tokens, 1)
        total_n_prompt_tokens += 1

    if _new_prompt_tokens is not None:
        ffi.release(_new_prompt_tokens)

    ffi.release(n_past)
