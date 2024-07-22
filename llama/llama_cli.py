__all__ = ['llama_generate']

from numba import cuda


if cuda.is_available():
    from .llama_cli_cffi_cuda_12_5 import llama_generate
else:
    from .llama_cli_cffi_cpu import llama_generate
