__all__ = ['llama_generate', 'LlamaOptions']

from ctypes import *
from queue import Queue
from typing import Iterator
from threading import Thread
from functools import partial

from huggingface_hub import hf_hub_download

from .llama_cli_options import LlamaOptions, convert_options_to_bytes


lib = CDLL('./llama/libllama-cli.so')
lib.llama_cli_main.argtypes = [c_int, POINTER(c_char_p)]
lib.llama_cli_main.restype = c_int

FPRINTF_FUNC = CFUNCTYPE(c_int, c_void_p, c_char_p, c_char_p)
FFLUSH_FUNC = CFUNCTYPE(c_int, c_void_p)


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

    lib.llama_set_fprintf(fprintf)
    lib.llama_set_fflush(fflush)

    argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(options)
    argc = len(argv)
    argv = (c_char_p * argc)(*argv)

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
