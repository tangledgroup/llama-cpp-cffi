__all__ = [
    'lib',
    'ffi',
    'lock',
    'global_weakkeydict',
    'llama_cpp_cffi_ggml_log_callback',
    'backend_init',
    'backend_free',
]

import os
os.environ['GGML_VK_DISABLE_COOPMAT'] = os.getenv('GGML_VK_DISABLE_COOPMAT', '1')
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'true')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.getenv('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

import atexit
from threading import Lock
from typing import TypeAlias
from weakref import WeakKeyDictionary

from .gpu import is_cuda_available, is_vulkan_available


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


global_weakkeydict = WeakKeyDictionary()


#
# low-level API
#
void_p: TypeAlias = ffi.typeof('void*') # type: ignore
char_p: TypeAlias = ffi.typeof('char*') # type: ignore
int_p: TypeAlias = ffi.typeof('int*') # type: ignore
float_p: TypeAlias = ffi.typeof('float*') # type: ignore
ggml_log_level: TypeAlias = ffi.typeof('enum ggml_log_level') # type: ignore
ggml_numa_strategy: TypeAlias = ffi.typeof('enum ggml_numa_strategy') # type: ignore
llama_model_params: TypeAlias = ffi.typeof('struct llama_model_params') # type: ignore
llama_model: TypeAlias = ffi.typeof('struct llama_model') # type: ignore
llama_model_p: TypeAlias = ffi.typeof('struct llama_model*') # type: ignore
llama_context: TypeAlias = ffi.typeof('struct llama_context') # type: ignore
llama_context_p: TypeAlias = ffi.typeof('struct llama_context*') # type: ignore
llama_context_params: TypeAlias = ffi.typeof('struct llama_context_params') # type: ignore
llama_sampler: TypeAlias = ffi.typeof('struct llama_sampler') # type: ignore
llama_sampler_p: TypeAlias = ffi.typeof('struct llama_sampler*') # type: ignore
llama_sampler_chain_params: TypeAlias = ffi.typeof('struct llama_sampler_chain_params') # type: ignore
llama_batch: TypeAlias = ffi.typeof('struct llama_batch') # type: ignore
llama_batch_p: TypeAlias = ffi.typeof('struct llama_batch*') # type: ignore
llama_pos: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_token: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_seq_id: TypeAlias = ffi.typeof('int32_t') # type: ignore
llama_token_data: TypeAlias = ffi.typeof('struct llama_token_data') # type: ignore
llama_token_data_p: TypeAlias = ffi.typeof('struct llama_token_data*') # type: ignore
llama_token_data_array: TypeAlias = ffi.typeof('struct llama_token_data_array') # type: ignore
llama_token_data_array_p: TypeAlias = ffi.typeof('struct llama_token_data_array*') # type: ignore
clip_ctx: TypeAlias = ffi.typeof('struct clip_ctx') # type: ignore
clip_ctx_p: TypeAlias = ffi.typeof('struct clip_ctx*') # type: ignore
clip_image_size: TypeAlias = ffi.typeof('struct clip_image_size') # type: ignore
clip_image_size_p: TypeAlias = ffi.typeof('struct clip_image_size*') # type: ignore
llava_image_embed: TypeAlias = ffi.typeof('struct llava_image_embed') # type: ignore
llava_image_embed_p: TypeAlias = ffi.typeof('struct llava_image_embed*') # type: ignore

lock = Lock()


# Set callback for all future logging events.
# If this is not called, or NULL is supplied, everything is output on stderr.
#
# LLAMA_API void llama_log_set(ggml_log_callback log_callback, void * user_data);
#
# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
@ffi.def_extern()
def llama_cpp_cffi_ggml_log_callback(level: ggml_log_level, text: char_p, user_data: void_p):
    pass


# disable logs by default
lib.llama_log_set(lib.llama_cpp_cffi_ggml_log_callback, ffi.NULL)


def backend_init():
    with lock:
        lib.llama_backend_init()


def backend_free():
    with lock:
        lib.llama_backend_free()


#
# backend
#
backend_init()
atexit.register(backend_free)
