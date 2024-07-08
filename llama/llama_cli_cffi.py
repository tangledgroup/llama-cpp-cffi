__all__ = ['llama_generate', 'LlamaOptions']

import ctypes
from queue import Queue
from typing import Iterator
from threading import Thread
from functools import partial

from huggingface_hub import hf_hub_download

from .llama_cli_options import LlamaOptions, convert_options_to_bytes
from ._llama_cli import lib, ffi

FPRINTF_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
FFLUSH_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


def fprintf_func(file_obj, fmt, *args, queue=None):
    content = fmt.decode('utf-8') % tuple(arg.decode('utf-8') for arg in args)
    queue.put(content)
    size = len(content)
    return size


def fflush_func(file_obj):
    return 0


def _llama_cli_main(argc, argv, queue):
    r = lib.llama_cli_main(argc, argv)
    assert r == 0
    queue.put(None)


def llama_generate(options: LlamaOptions) -> Iterator[str]:
    # check hf_repo, hf_file
    if options.hf_repo and options.hf_file:
        options.model = hf_hub_download(repo_id=options.hf_repo, filename=options.hf_file)
        options.hf_repo = None
        options.hf_file = None
    elif options.model:
        pass
    else:
        raise ValueError(f'hf_repo = {options.hf_repo}, hf_file = {options.hf_file}')

    assert options.model

    queue = Queue()

    fprintf = FPRINTF_FUNC(partial(fprintf_func, queue=queue))
    fflush = FFLUSH_FUNC(fflush_func)

    fprintf_address = ctypes.cast(fprintf, ctypes.c_void_p).value
    fflush_address = ctypes.cast(fflush, ctypes.c_void_p).value

    cffi_fprintf_callback = ffi.cast('int (*func)(FILE*, const char* format, ...)', fprintf_address)
    cffi_fflush_callback = ffi.cast('int (*func)(FILE*)', fflush_address)

    lib.llama_set_fprintf(cffi_fprintf_callback)
    lib.llama_set_fflush(cffi_fflush_callback)

    argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(options)
    argv = [ffi.new('char[]', n) for n in argv]
    argc = len(argv)

    t = Thread(target=_llama_cli_main, args=(argc, argv, queue))
    t.start()

    while True:
        chunk = queue.get()
        queue.task_done()

        if chunk is None:
            break

        yield chunk

    queue.join()
    t.join()
