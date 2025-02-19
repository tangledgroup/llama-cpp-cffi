__all__ = ['Model']

import gc
from typing import Optional, Iterator

from attrs import define, field, asdict
from transformers import AutoConfig
from huggingface_hub import hf_hub_download

from .formatter import get_config
from .options import ModelOptions, CompletionsOptions
from .llama_cpp import lib, ffi, lock, llama_model_p, llama_model_params
from .llava import llava_completions
from .minicpmv import minicpmv_completions
from .qwen2vl import qwen2vl_completions
from .text import text_completions


def model_init(model_options: ModelOptions) -> llama_model_p:
    model_path = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.hf_file)
    # print(f'{model_path=}')

    model_params: llama_model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = model_options.gpu_layers
    # model_params.split_mode = model_options.split_mode # FIXME: check Options
    model_params.main_gpu = model_options.main_gpu
    model_params.use_mlock = model_options.use_mlock
    model_params.use_mmap = model_options.use_mmap
    model_params.check_tensors = model_options.check_tensors

    with lock:
        model: llama_model_p = lib.llama_model_load_from_file(model_path.encode(), model_params)

    # assert model != ffi.NULL

    if model == ffi.NULL:
        raise MemoryError(f'Could not init model: {model_options=} {model=}')

    return model


def model_free(model: llama_model_p):
    with lock:
        lib.llama_model_free(model)


@define
class Model:
    options: Optional[ModelOptions] = None
    _model: Optional[llama_model_p] = field(default=None, eq=False) # C object


    def __init__(self,
                 creator_hf_repo: Optional[str]=None,
                 hf_repo: Optional[str]=None,
                 hf_file: Optional[str]=None,
                 mmproj_hf_file: Optional[str]=None,
                 tokenizer_hf_repo: Optional[str]=None,
                 options: Optional[ModelOptions]=None):
        if not options:
            options = ModelOptions()

        if creator_hf_repo:
            options.creator_hf_repo = creator_hf_repo
            options.hf_repo = hf_repo
            options.hf_file = hf_file
            options.mmproj_hf_file = mmproj_hf_file
            options.tokenizer_hf_repo = tokenizer_hf_repo

        self.__attrs_init__(options) # type: ignore


    def __str__(self) -> str:
        if not self.options:
            return ''

        return ':'.join([
            self.options.creator_hf_repo or '',
            self.options.hf_repo or '',
            self.options.hf_file or '',
            self.options.mmproj_hf_file or '',
            self.options.tokenizer_hf_repo or '',
        ])


    @classmethod
    def from_str(cls, s: str) -> 'Model':
        model: Model = Model(*s.split(':'))
        return model


    def init(self, **kwargs):
        self.options = ModelOptions(**(asdict(self.options) | kwargs))

        # self._model = model_init(self.options)
        #
        # if self._model == ffi.NULL:
        #     raise MemoryError(f'Could not load model: {self.options}')
        self._model = model_init(self.options)

        print(f'Model.init {self._model=}')


    def free(self):
        print(f'Model.free {self._model=}')
        self.options = None

        if self._model:
            model_free(self._model)
            self._model = None
            gc.collect()


    def completions(self, **kwargs) -> Iterator[str]:
        assert self.options
        assert self._model

        config: AutoConfig = get_config(self.options.creator_hf_repo)
        model_type: str = config.model_type # type: ignore
        # print(f'{model_type=}')

        if (
            model_type.startswith('llava') or
            model_type.startswith('bunny') or
            model_type.startswith('moondream')
        ):
            completions_func = llava_completions
        elif 'minicpmv' in model_type:
            completions_func = minicpmv_completions
        elif 'qwen2_vl' in model_type:
            completions_func = qwen2vl_completions
        else:
            completions_func = text_completions

        model_options: ModelOptions = self.options
        completions_options = CompletionsOptions(**kwargs)

        # used for stop
        last_n_tokens: int = 32
        last_n_tokens_buffer: list = []

        # print(f'{model_options=}')
        # print(f'{completions_options=}')

        for token in completions_func(self, model_options, completions_options):
            yield token

            # check if should stop
            if not completions_options.stop:
                continue

            if len(last_n_tokens_buffer) >= last_n_tokens:
                last_n_tokens_buffer = last_n_tokens_buffer[1:]

            last_n_tokens_buffer.append(token)

            try:
                buffer = ''.join(last_n_tokens_buffer)
            except Exception:
                continue

            if completions_options.stop in buffer:
                print('[BREAK]')
                break
