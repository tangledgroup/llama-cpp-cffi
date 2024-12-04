__all__ = [
    # high-level API
    'completions',

    # low-level API
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
    'text_completions',
    'clip_completions',
    'mllama_completions',
]

import os
import ctypes
from queue import Queue
from copy import deepcopy
from typing import Any, Optional, Iterator, Callable, NewType
from threading import Thread
from functools import partial

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from .model import Model
from .options import Options, convert_options_to_bytes
from .util import is_cuda_available, is_vulkan_available
from .formatter import get_tokenizer, get_special_tokens, format_messages


LLAMA_CPP_BACKEND = os.getenv('LLAMA_CPP_BACKEND', None)


try:
    if LLAMA_CPP_BACKEND:
        if LLAMA_CPP_BACKEND in ('cuda', 'CUDA'):
            from ._llama_cli_cuda_12_6_3 import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_cuda_12_6_3 import lib as llava_lib, llava_ffi
            from ._minicpmv_cli_cuda_12_6_3 import lib as minicpmv_lib, ffi as minicpmv_ffi
        elif LLAMA_CPP_BACKEND in ('vulkan', 'VULKAN'):
            from ._llama_cli_vulkan_1_x import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_vulkan_1_x import lib as llava_lib, ffi as llava_ffi
            from ._minicpmv_cli_vulkan_1_x import lib as minicpmv_lib, ffi as minicpmv_ffi
        elif LLAMA_CPP_BACKEND in ('cpu', 'CPU'):
            from ._llama_cli_cpu import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_cpu import lib as llava_lib, ffi as llava_ffi
            from ._minicpmv_cli_cpu import lib as minicpmv_lib, ffi as minicpmv_ffi
        else:
            raise ValueError(f'{LLAMA_CPP_BACKEND = }')
    else:
        if is_cuda_available():
            from ._llama_cli_cuda_12_6_3 import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_cuda_12_6_3 import lib as llava_lib, ffi as llava_ffi
            from ._minicpmv_cli_cuda_12_6_3 import lib as minicpmv_lib, ffi as minicpmv_ffi
        elif is_vulkan_available():
            from ._llama_cli_vulkan_1_x import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_vulkan_1_x import lib as llava_lib, ffi as llava_ffi
            from ._minicpmv_cli_vulkan_1_x import lib as minicpmv_lib, ffi as minicpmv_ffi
        else:
            from ._llama_cli_cpu import lib as llama_lib, ffi as llama_ffi
            from ._llava_cli_cpu import lib as llava_lib, ffi as llava_ffi
            from ._minicpmv_cli_cpu import lib as minicpmv_lib, ffi as minicpmv_ffi
except ImportError:
    from ._llama_cli_cpu import lib as llama_lib, ffi as llama_ffi
    from ._llava_cli_cpu import lib as llava_lib, ffi as llava_ffi
    from ._minicpmv_cli_cpu import lib as minicpmv_lib, ffi as minicpmv_ffi


_LLAMA_YIELD_TOKEN_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
_LLAMA_SHOULD_STOP_T = ctypes.CFUNCTYPE(ctypes.c_int)


def _llama_yield_token_func(chunk_bytes: bytes, queue: Queue, metadata: dict):
    stop_on_special_token = metadata['stop_on_special_token']
    special_tokens = metadata['special_tokens']

    try:
        b: bytes = metadata['prev_chunk_bytes'] + chunk_bytes
        chunk = b.decode()
    except UnicodeDecodeError:
        metadata['prev_chunk_bytes'] += chunk_bytes
        return

    metadata['prev_chunk_bytes'] = b''

    if not stop_on_special_token:
        queue.put(chunk)
        return

    # detect stop token
    buffer = metadata['buffer']
    buffer += chunk
    metadata['buffer'] = buffer

    subtoken_found = False
    token_found = False

    for token in special_tokens:
        for i in range(len(token)):
            subtoken = token[:i + 1]

            if buffer[-len(subtoken):] == subtoken:
                subtoken_found = True

                if token in buffer:
                    index = buffer.index(token)
                    chunk = buffer[:index]
                    buffer = buffer[index + len(token):]
                    metadata['buffer'] = buffer
                    metadata['should_stop'] = True
                    token_found = True
                    break

        if subtoken_found or token_found:
            break

    if subtoken_found:
        return

    if token_found:
        return

    buffer = metadata['buffer']
    queue.put(buffer)
    metadata['buffer'] = ''


def _llama_should_stop_func(queue: Queue, metadata: dict) -> int:
    return 1 if metadata['should_stop'] else 0


def _llama_cli_main(argc, argv, queue: Queue, metadata: dict):
    _llama_yield_token = _LLAMA_YIELD_TOKEN_T(partial(_llama_yield_token_func, queue=queue, metadata=metadata))
    _llama_should_stop = _LLAMA_SHOULD_STOP_T(partial(_llama_should_stop_func, queue=queue, metadata=metadata))

    _llama_yield_token_address = ctypes.cast(_llama_yield_token, ctypes.c_void_p).value
    _llama_should_stop_address = ctypes.cast(_llama_should_stop, ctypes.c_void_p).value

    cffi__llama_yield_token_callback = llama_ffi.cast('void (*_llama_yield_token_t)(const char * token)', _llama_yield_token_address)
    cffi__llama_should_stop_callback = llama_ffi.cast('int (*_llama_should_stop_t)(void)', _llama_should_stop_address)

    r = llama_lib._llama_cli_main(argc, argv, cffi__llama_yield_token_callback, cffi__llama_should_stop_callback)
    # assert r == 0
    queue.put(None)


def _llava_cli_main(argc, argv, queue: Queue, metadata: dict):
    _llama_yield_token = _LLAMA_YIELD_TOKEN_T(partial(_llama_yield_token_func, queue=queue, metadata=metadata))
    _llama_should_stop = _LLAMA_SHOULD_STOP_T(partial(_llama_should_stop_func, queue=queue, metadata=metadata))

    _llama_yield_token_address = ctypes.cast(_llama_yield_token, ctypes.c_void_p).value
    _llama_should_stop_address = ctypes.cast(_llama_should_stop, ctypes.c_void_p).value

    cffi__llama_yield_token_callback = llama_ffi.cast('void (*_llama_yield_token_t)(const char * token)', _llama_yield_token_address)
    cffi__llama_should_stop_callback = llama_ffi.cast('int (*_llama_should_stop_t)(void)', _llama_should_stop_address)

    r = llava_lib._llava_cli_main(argc, argv, cffi__llama_yield_token_callback, cffi__llama_should_stop_callback)
    # assert r == 0
    queue.put(None)


def _minicpmv_cli_main(argc, argv, queue: Queue, metadata: dict):
    _llama_yield_token = _LLAMA_YIELD_TOKEN_T(partial(_llama_yield_token_func, queue=queue, metadata=metadata))
    _llama_should_stop = _LLAMA_SHOULD_STOP_T(partial(_llama_should_stop_func, queue=queue, metadata=metadata))

    _llama_yield_token_address = ctypes.cast(_llama_yield_token, ctypes.c_void_p).value
    _llama_should_stop_address = ctypes.cast(_llama_should_stop, ctypes.c_void_p).value

    cffi__llama_yield_token_callback = llama_ffi.cast('void (*_llama_yield_token_t)(const char * token)', _llama_yield_token_address)
    cffi__llama_should_stop_callback = llama_ffi.cast('int (*_llama_should_stop_t)(void)', _llama_should_stop_address)

    r = minicpmv_lib._minicpmv_cli_main(argc, argv, cffi__llama_yield_token_callback, cffi__llama_should_stop_callback)
    # assert r == 0
    queue.put(None)


def completions(options: Options) -> Iterator[str]:
    tokenizer: AutoTokenizer
    creator_hf_repo: str
    prompt: str
    queue: Queue

    assert options.model and isinstance(options.model, Model)

    options: Options = deepcopy(options)

    engine = options.engine
    engine_func: str = f'_{engine}_cli_main'
    engine_func: Callable = globals()[engine_func]

    model: Model = options.model

    if model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(model.creator_hf_repo)

    if model.mmproj_hf_file:
        options.mmproj = hf_hub_download(repo_id=model.hf_repo, filename=model.mmproj_hf_file)

    options.model = hf_hub_download(repo_id=model.hf_repo, filename=model.hf_file)

    if isinstance(options.prompt, list):
        options.prompt = format_messages(tokenizer, options.prompt, options)

    if options.no_display_prompt == False:
        # print('options.prompt:')
        print(options.prompt, end='')

    if options.no_display_prompt == False:
        options.no_display_prompt = None

    if options.log_disable == False:
        options.log_disable = None

    queue = Queue()
    special_tokens: list[str] = get_special_tokens(tokenizer, force_standard_special_tokens=True)

    metadata: dict = {
        'prev_chunk_bytes': b'',
        'buffer': '',
        'stop_on_special_token': True,
        'should_stop': False,
        'special_tokens': special_tokens,
    }

    if isinstance(options.stop, str):
        metadata['special_tokens'].append(options.stop)
    elif isinstance(options.stop, (list, tuple)):
        metadata['special_tokens'].extend(list(options.stop))
    elif options.stop is not None:
        raise ValueError(options.stop)

    argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(options)
    argv = [llama_ffi.new('char[]', n) for n in argv]
    argc = len(argv)

    t = Thread(target=engine_func, args=(argc, argv, queue, metadata))
    t.start()

    try:
        while True:
            chunk = queue.get()
            queue.task_done()

            if chunk is None:
                break

            yield chunk
    except GeneratorExit:
        # give signal to thread to stop
        metadata['should_stop'] = True

    queue.join()
    t.join()


#
# low-level API
#
lib, ffi = llama_lib, llama_ffi

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
llava_image_embed  = NewType('llava_image_embed', ffi.typeof('struct llava_image_embed'))
llava_image_embed_p  = NewType('llava_image_embed*', ffi.typeof('struct llava_image_embed*'))

LLAMA_DEFAULT_SEED = 0xFFFFFFFF


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


def sampler_init(options: Options) -> llama_sampler_p:
    sampler_params: llama_sampler_chain_params = lib.llama_sampler_chain_default_params()
    sampler: llama_sampler_p = lib.llama_sampler_chain_init(sampler_params)
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_k(options.top_k))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_top_p(options.top_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_min_p(options.min_p, 1))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_temp(options.temp))
    lib.llama_sampler_chain_add(sampler, lib.llama_sampler_init_dist(LLAMA_DEFAULT_SEED))
    return sampler


def sampler_free(sampler: llama_sampler_p):
    lib.llama_sampler_free(sampler)


def clip_init_context(options: Options) -> clip_ctx_p:
    clip_path: str = hf_hub_download(repo_id=options.model.hf_repo, filename=options.model.mmproj_hf_file)
    ctx_clip: clip_ctx_p = lib.clip_model_load(clip_path.encode(), 1)
    return ctx_clip


def clip_free_context(ctx_clip: clip_ctx_p):
    lib.clip_free(ctx_clip)


def check_context_size(context: llama_context_p, batch: llama_batch) -> int:
    n_ctx: int = lib.llama_n_ctx(context)
    n_ctx_used: int = lib.llama_get_kv_cache_used_cells(context)

    if n_ctx_used + batch.n_tokens > n_ctx:
        return 1

    return 0


#
# llm
#
def text_completions(model: llama_model_p,
                     context: llama_context_p,
                     sampler: llama_sampler_p,
                     options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str)

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

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
# vlm
#
def batch_get_one_and_decode(context: llama_context_p, prompt: str, n_batch: int, n_past: Optional[int_p], tokenizer: AutoTokenizer) -> Optional[llama_batch]:
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


def process_eval_image_embed(context: llama_context_p, clip_context: clip_ctx_p, embeds: llava_image_embed_p, n_batch: int, n_past: int_p, idx: int):
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
                     clip_context: clip_ctx_p,
                     options: Options) -> Iterator[str]:
    assert isinstance(options.prompt, str)
    assert isinstance(options.image, str)

    # tokenizer
    tokenizer: AutoTokenizer

    if options.model.tokenizer_hf_repo:
        tokenizer = get_tokenizer(options.model.tokenizer_hf_repo)
    else:
        tokenizer = get_tokenizer(options.model.creator_hf_repo)

    # llava - process image
    n_past: int_p = ffi.new('int*', 0)
    embeds: llava_image_embed_p = lib.llava_image_embed_make_with_filename(clip_context, options.threads, options.image.encode())
    idx: int = 0
    num_image_embeds = embeds.n_image_pos / lib.clip_n_patches(clip_context)
    has_minicpmv_projector = lib.clip_is_minicpmv(clip_context)
    print(f'{has_minicpmv_projector=}')

    messages = [{'role': 'user', 'content': '<image>'}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt = prompt[:prompt.index('<image>') + len('<image>')]
    print('prompt [0]:', prompt)

    batch = batch_get_one_and_decode(context, prompt, options.batch_size, n_past, tokenizer)
    process_eval_image_embed(context, clip_context, embeds, options.batch_size, n_past, idx)
    idx += 1
    batch = batch_get_one_and_decode(context, '</image>', options.batch_size, n_past, tokenizer)

    if has_minicpmv_projector >= 2 and num_image_embeds > 1:
        num_image_embeds_col = lib.clip_uhd_num_image_embeds_col(clip_context)
        batch = batch_get_one_and_decode(context, '<slice>', options.batch_size, n_past, tokenizer)
        i = 0

        while i < (num_image_embeds - 1) / num_image_embeds_col:
            j = 0

            while j < num_image_embeds_col:
                batch = batch_get_one_and_decode(context, '<image>', options.batch_size, n_past, tokenizer)
                process_eval_image_embed(context, clip_context, embeds, options.batch_size, n_past, idx)
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

        if lib.llama_decode(context, batch):
            break

        new_token_id = lib.llama_sampler_sample(sampler, context, -1)

        if lib.llama_token_is_eog(model, new_token_id):
            break

        piece: str = tokenizer.decode(new_token_id)
        yield piece

        # next batch
        if _new_prompt_tokens is not None:
            ffi.release(_new_prompt_tokens)

        _new_prompt_tokens = ffi.new('llama_token[]', [new_token_id])
        batch = lib.llama_batch_get_one(_new_prompt_tokens, 1)
        total_n_prompt_tokens += 1

    if _new_prompt_tokens is not None:
        ffi.release(_new_prompt_tokens)

    ffi.release(n_past)


mllama_completions = None
