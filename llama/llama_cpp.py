__all__ = [
    'lib',
    'ffi',
    'lock',
    'global_weakkeydict',
    'llama_cpp_cffi_ggml_log_callback',
    'backend_init',
    'backend_free',
    'void_p',
    'char_p',
    'int_p',
    'float_p',
    'ggml_log_level',
    'ggml_numa_strategy',
    'llama_model_params',
    'llama_model',
    'llama_model_p',
    'llama_context',
    'llama_context_p',
    'llama_context_params',
    'llama_sampler',
    'llama_sampler_p',
    'llama_sampler_chain_params',
    'llama_batch',
    'llama_batch_p',
    'llama_pos',
    'llama_token',
    'llama_seq_id',
    'llama_token_data',
    'llama_token_data_p',
    'llama_token_data_array',
    'llama_token_data_array_p',
    'llama_vocab',
    'llama_vocab_p',
    'clip_ctx',
    'clip_ctx_p',
    'clip_image_size',
    'clip_image_size_p',
    'llava_image_embed',
    'llava_image_embed_p',
    'ggml_type',
]

import os
os.environ['GGML_VK_DISABLE_COOPMAT'] = os.getenv('GGML_VK_DISABLE_COOPMAT', '1')
os.environ['TOKENIZERS_PARALLELISM'] = os.getenv('TOKENIZERS_PARALLELISM', 'true')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = os.getenv('TRANSFORMERS_NO_ADVISORY_WARNINGS', '1')

import atexit
from enum import Enum
from threading import Lock
from typing import TypeAlias
from weakref import WeakKeyDictionary

from .gpu import is_cuda_available, is_vulkan_available


LLAMA_CPP_BACKEND = os.getenv('LLAMA_CPP_BACKEND', None)


try:
    if LLAMA_CPP_BACKEND:
        if LLAMA_CPP_BACKEND in ('cuda', 'CUDA'):
            from ._llama_cpp_cuda_12_6_3 import lib, ffi # type: ignore
        elif LLAMA_CPP_BACKEND in ('vulkan', 'VULKAN'):
            from ._llama_cpp_vulkan_1_x import lib, ffi # type: ignore
        elif LLAMA_CPP_BACKEND in ('cpu', 'CPU'):
            from ._llama_cpp_cpu import lib, ffi # type: ignore
        else:
            raise ValueError(f'{LLAMA_CPP_BACKEND = }')
    else:
        if is_cuda_available():
            from ._llama_cpp_cuda_12_6_3 import lib, ffi # type: ignore
        elif is_vulkan_available():
            from ._llama_cpp_vulkan_1_x import lib, ffi # type: ignore
        else:
            from ._llama_cpp_cpu import lib, ffi # type: ignore
except ImportError:
    from ._llama_cpp_cpu import lib, ffi # type: ignore


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
llama_vocab: TypeAlias = ffi.typeof('struct llama_vocab') # type: ignore
llama_vocab_p: TypeAlias = ffi.typeof('struct llama_vocab*') # type: ignore
clip_ctx: TypeAlias = ffi.typeof('struct clip_ctx') # type: ignore
clip_ctx_p: TypeAlias = ffi.typeof('struct clip_ctx*') # type: ignore
clip_image_size: TypeAlias = ffi.typeof('struct clip_image_size') # type: ignore
clip_image_size_p: TypeAlias = ffi.typeof('struct clip_image_size*') # type: ignore
llava_image_embed: TypeAlias = ffi.typeof('struct llava_image_embed') # type: ignore
llava_image_embed_p: TypeAlias = ffi.typeof('struct llava_image_embed*') # type: ignore


class ggml_type(Enum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    # Q4_2 = 4 support has been removed
    # Q4_3 = 5 support has been removed
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    # Q4_0_4_4 = 31 support has been removed from gguf files
    # Q4_0_4_8 = 32
    # Q4_0_8_8 = 33
    TQ1_0   = 34
    TQ2_0   = 35
    # IQ4_NL_4_4 = 36
    # IQ4_NL_4_8 = 37
    # IQ4_NL_8_8 = 38
    # COUNT   = 39

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
