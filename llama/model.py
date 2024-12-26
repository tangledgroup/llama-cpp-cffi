__all__ = ['Model']

from typing import Optional, Unpack, Iterator

from attrs import define, asdict
from transformers import AutoConfig

from .options import ModelOptions, CompletionsOptions

from .llama import (
    llama_model_p,
    model_init,
    model_free,
    # context_init,
    # context_free,
    text_completions,
    clip_completions,
    # mllama_completions,
)

from .formatter import get_config


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
        # print(f'{model_type=}')

        if 'llava' in model_type or 'moondream' in model_type or 'minicpmv' in model_type or 'qwen2_vl' in model_type:
            completions_func = clip_completions
        else:
            completions_func = text_completions

        model_options: ModelOptions = self.options
        completions_options = CompletionsOptions(**kwargs)

        for token in completions_func(self._model, model_options, completions_options):
            yield token
