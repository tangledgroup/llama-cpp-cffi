__all__ = ['Model']

from typing import Optional, Iterator

from attrs import define, asdict
from transformers import AutoConfig
from huggingface_hub import hf_hub_download

from .formatter import get_config
from .options import ModelOptions, CompletionsOptions
from .llama_cpp import lib, ffi, lock, llama_model_p, llama_model_params
from .llava import llava_completions
from .qwen2vl import qwen2vl_completions
from .text import text_completions


def model_init(model_options: ModelOptions) -> llama_model_p:
    model_path = hf_hub_download(repo_id=model_options.hf_repo, filename=model_options.hf_file)
    print(f'{model_path=}')

    model_params: llama_model_params = lib.llama_model_default_params()
    model_params.n_gpu_layers = model_options.gpu_layers
    # model_params.split_mode = model_options.split_mode # FIXME: check Options
    model_params.main_gpu = model_options.main_gpu
    model_params.use_mmap = not model_options.no_mmap # TODO: use exact field names like in structs/API
    model_params.use_mlock = model_options.mlock
    model_params.check_tensors = model_options.check_tensors

    with lock:
        model: llama_model_p = lib.llama_load_model_from_file(model_path.encode(), model_params)

    return model


def model_free(model: llama_model_p):
    with lock:
        lib.llama_free_model(model)


@define
class Model:
    options: Optional[ModelOptions] = None
    _model: Optional[llama_model_p] = None


    def __init__(self,
                 creator_hf_repo: str,
                 hf_repo: str,
                 hf_file: str,
                 mmproj_hf_file: Optional[str]=None,
                 tokenizer_hf_repo: Optional[str]=None):
        options = ModelOptions(
            creator_hf_repo=creator_hf_repo,
            hf_repo=hf_repo,
            hf_file=hf_file,
            mmproj_hf_file=mmproj_hf_file,
            tokenizer_hf_repo=tokenizer_hf_repo,
        )

        self.__attrs_init__(options) # type: ignore


    def __del__(self):
        self.options = None

        if self._model:
            model_free(self._model)
            self._model = None


    def init(self, **kwargs):
        self.options = ModelOptions(**(asdict(self.options) | kwargs))
        self._model = model_init(self.options)


    def free(self):
        self.options = None

        if self._model:
            model_free(self._model)
            self._model = None


    def completions(self, **kwargs) -> Iterator[str]:
        assert self.options
        assert self._model

        config: AutoConfig = get_config(self.options.creator_hf_repo)
        model_type: str = config.model_type # type: ignore
        print(f'{model_type=}')

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

        for token in completions_func(self, model_options, completions_options):
            yield token
