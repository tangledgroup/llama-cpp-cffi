__all__ = ['llama_generate', 'Options']

import json
import ctypes
from queue import Queue
from copy import deepcopy
from typing import Iterator
from threading import Thread
from functools import partial

from huggingface_hub import hf_hub_download

from .llama_cli_options import Options, convert_options_to_bytes
from ._llama_cli import lib, ffi


FPRINTF_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
FFLUSH_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)




def _llama_cli_main(argc, argv, queue=None, callback=None, metadata=None):
    r = lib.llama_cli_main(argc, argv)
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

    # get bos, eos, and eot from metedata
    metadata_options = deepcopy(options)
    metadata_options.log_disable = True
    metadata_argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(metadata_options)
    metadata_argv = [ffi.new('char[]', n) for n in metadata_argv]
    metadata_argc = len(metadata_argv)

    c_metadata: 'const char*' = lib.llama_get_metadata_as_json(metadata_argc, metadata_argv)
    metadata: bytes = ffi.string(c_metadata)
    lib.llama_free_metadata_as_json(c_metadata)
    metadata: str = metadata.decode('utf-8')
    metadata: dict = json.loads(metadata)
    print(f'{metadata = }')

    # intercept token generation
    fprintf = FPRINTF_FUNC(partial(fprintf_func, queue=queue, metadata=metadata))
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
