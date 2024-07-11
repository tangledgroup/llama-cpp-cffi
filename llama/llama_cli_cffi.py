__all__ = ['llama_generate', 'Options']

import json
import ctypes
from queue import Queue
from typing import Iterator
from threading import Thread
from functools import partial

from huggingface_hub import hf_hub_download

from .llama_cli_model import Model
from .llama_cli_options import Options, convert_options_to_bytes
from ._llama_cli import lib, ffi


_LLAMA_YIELD_TOKEN_T = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
_LLAMA_SHOULD_STOP_T = ctypes.CFUNCTYPE(ctypes.c_int)


def _llama_yield_token_func(chunk: bytes, queue=None, callback=None, metadata=None):
    chunk = chunk.decode()
    print(chunk, flush=True, end='')


def _llama_should_stop_func(queue=None, callback=None, metadata=None) -> int:
    return 0


def _llama_cli_main(argc, argv, queue=None, callback=None, metadata=None):
    _llama_yield_token = _LLAMA_YIELD_TOKEN_T(partial(_llama_yield_token_func, queue=queue, callback=callback, metadata=metadata))
    _llama_should_stop = _LLAMA_SHOULD_STOP_T(partial(_llama_should_stop_func, queue=queue, callback=callback, metadata=metadata))

    _llama_yield_token_address = ctypes.cast(_llama_yield_token, ctypes.c_void_p).value
    _llama_should_stop_address = ctypes.cast(_llama_should_stop, ctypes.c_void_p).value

    cffi__llama_yield_token_callback = ffi.cast('void (*_llama_yield_token_t)(const char * token)', _llama_yield_token_address)
    cffi__llama_should_stop_callback = ffi.cast('int (*_llama_should_stop_t)(void)', _llama_should_stop_address)

    r = lib._llama_cli_main(argc, argv, cffi__llama_yield_token_callback, cffi__llama_should_stop_callback, 1)
    assert r == 0
    
    if queue is not None:
        queue.put(None)
    elif callback is not None:
        callback(None)


def llama_generate(options: Options, callback=None) -> Iterator[str] | None:
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

    if callback:
        queue = None
    else:
        queue = Queue()

    metadata: dict = {}

    argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(options)
    argv = [ffi.new('char[]', n) for n in argv]
    argc = len(argv)

    if callback:
        _llama_cli_main(argc, argv, queue, callback, metadata)
        yield ''
    else:
        t = Thread(target=_llama_cli_main, args=(argc, argv, queue, callback, metadata))
        t.start()

        while True:
            chunk = queue.get()
            queue.task_done()

            if chunk is None:
                break

            yield chunk

        queue.join()
        t.join()
