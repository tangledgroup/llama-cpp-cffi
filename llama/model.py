__all__ = ['Model']

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
        model: llama_model_p = lib.llama_load_model_from_file(model_path.encode(), model_params)

    return model


def model_free(model: llama_model_p):
    with lock:
        lib.llama_free_model(model)


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


    def __del__(self):
        self.options = None

        if self._model:
            model_free(self._model)
            self._model = None


    @classmethod
    def from_str(cls, s: str) -> 'Model':
        model: Model = Model(*s.split(':'))
        return model


    # def are_models_defs_equal(self, other: 'Model') -> bool:
    #     if not self.options or not other.options:
    #         return False

    #     return all([
    #         self.options.creator_hf_repo == other.options.creator_hf_repo,
    #         self.options.hf_repo == other.options.hf_repo,
    #         self.options.hf_file == other.options.hf_file,
    #         self.options.mmproj_hf_file == other.options.mmproj_hf_file,
    #         self.options.tokenizer_hf_repo == other.options.tokenizer_hf_repo,
    #     ])


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

        for token in completions_func(self, model_options, completions_options):
            yield token
