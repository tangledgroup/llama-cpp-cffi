__all__ = ['llama_generate', 'LlamaOptions']

import os
import json
from ctypes import *
from queue import Queue
from copy import deepcopy
from typing import Iterator
from threading import Thread
from functools import partial

from huggingface_hub import hf_hub_download

from .llama_cli_options import LlamaOptions, convert_options_to_bytes

current_module_path = os.path.abspath(__file__)
current_module_dir = os.path.dirname(current_module_path)
libllama_cli_path = os.path.join(current_module_dir, 'libllama-cli.so')

lib = CDLL(libllama_cli_path)

lib.llama_cli_main.argtypes = [c_int, POINTER(c_char_p)]
lib.llama_cli_main.restype = c_int

lib.llama_get_metadata_as_json.argtypes = [c_int, POINTER(c_char_p)]
lib.llama_get_metadata_as_json.restype = c_void_p

lib.llama_free_metadata_as_json.argtypes = [c_void_p]
lib.llama_free_metadata_as_json.restype = None

FPRINTF_FUNC = CFUNCTYPE(c_int, c_void_p, c_char_p, c_char_p)
FFLUSH_FUNC = CFUNCTYPE(c_int, c_void_p)


def fprintf_func(file_obj, fmt, arg, queue=None, callback=None, metadata=None):
    content = arg.decode('utf-8')

    if metadata:
        eos = metadata.get('eos')
        eot = metadata.get('eot')

        if eos is not None and eos in content:
            content = content[:content.index(eos)]
        elif eot is not None and eot in content:
            content = content[:content.index(eot)]

    if queue is not None:
        if content:
            queue.put(content)
        else:
            queue.put(None)
    elif callback is not None:
        callback(content)

    size = len(content)
    return size


def fflush_func(file_obj):
    return 0


def _llama_cli_main(argc, argv, queue=None, callback=None, metadata=None):
    r = lib.llama_cli_main(argc, argv)
    assert r == 0

    if queue is not None:
        queue.put(None)
    elif callback is not None:
        callback(None)


def llama_generate(options: LlamaOptions, callback=None, metadata=None) -> Iterator[str] | None:
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
    metadata_argc = len(metadata_argv)
    metadata_argv = (c_char_p * metadata_argc)(*metadata_argv)

    c_metadata: 'const char*' = lib.llama_get_metadata_as_json(metadata_argc, metadata_argv)
    metadata: str = string_at(c_metadata)
    lib.llama_free_metadata_as_json(c_metadata)
    metadata: dict = json.loads(metadata)
    print(f'{metadata = }')

    # intercept token generation
    fprintf = FPRINTF_FUNC(partial(fprintf_func, queue=queue, callback=callback, metadata=metadata))
    fflush = FFLUSH_FUNC(fflush_func)

    lib.llama_set_fprintf(fprintf)
    lib.llama_set_fflush(fflush)

    argv: list[bytes] = [b'llama-cli'] + convert_options_to_bytes(options)
    argc = len(argv)
    argv = (c_char_p * argc)(*argv)

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
