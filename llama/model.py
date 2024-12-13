__all__ = ['Model']

from typing import Any, Optional, Unpack, Iterator

from attrs import define, asdict

from .options import Options

from .llama import (
    llama_model_p,
    model_init,
    model_free,
    context_init,
    context_free,
    text_completions,
    clip_completions,
    # mllama_completions,
)

from .formatter import get_config

@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str
    mmproj_hf_file: Optional[str] = None
    tokenizer_hf_repo: Optional[str] = None
    _model: Optional[llama_model_p] = None
    _options: Options = Options()


    def __str__(self):
        return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.mmproj_hf_file or ""}:{self.tokenizer_hf_repo or ""}'


    def __del__(self):
        if self._model:
            model_free(self._model)
            self._model = None


    def init(self, **options: Unpack[Options]):
        options = Options(
            **options,
            model=self,
        )

        self._model = model_init(options)
        self._options = options


    def free(self):
        if self._model:
            model_free(self._model)
            self._model = None


    def completions(self, **options: Unpack[Options]) -> Iterator[str]:
        options = Options(
            **(asdict(self._options, recurse=False) | options),
        )

        _model = self._model
        config = get_config(self.creator_hf_repo)
        model_type = config.model_type

        if 'llava' in model_type or 'moondream' in model_type or 'minicpmv' in model_type:
            completions_func = clip_completions
        else:
            completions_func = text_completions

        for token in completions_func(_model, options):
            yield token
