__all__ = ['Model']

from typing import Any, Optional, Unpack, Iterator

from attrs import define, asdict

from .options import Options

from .llama import (
    backend_init,
    backend_free,
    model_init,
    model_free,
    context_init,
    context_free,
    sampler_init,
    sampler_free,
    clip_init_context,
    clip_free_context,
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
    mmproj_hf_file: Optional[str] = ''
    tokenizer_hf_repo: Optional[str] = ''
    _model: Any = None
    _clip_context: Any = None
    _options: Options = Options()


    def __str__(self):
        return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.mmproj_hf_file}:{self.tokenizer_hf_repo}'


    def __del__(self):
        if self._clip_context:
            clip_free_context(self._clip_context)
            self._clip_context = None

        if self._model:
            model_free(self._model)
            self._model = None


    def init(self, **options: Unpack[Options]):
        # print(f'! {options=}')
        options = Options(
            **options,
            model=self,
        )

        self._model = model_init(options)

        config = get_config(self.creator_hf_repo)
        model_type = config.model_type

        if 'llava' in model_type or 'moondream' in model_type or 'minicpmv' in model_type:
            self._clip_context = clip_init_context(options)

        self._options = options


    def completions(self, **options: Unpack[Options]) -> Iterator[str]:
        options = Options(
            **(asdict(self._options, recurse=False) | options),
        )

        _context = context_init(self._model, options)
        _sampler = sampler_init(self._model, _context, options)

        if self._clip_context:
            for token in clip_completions(self._model, _context, _sampler, self._clip_context, options):
                yield token
        else:
            for token in text_completions(self._model, _context, _sampler, options):
                yield token

        sampler_free(_sampler)
        context_free(_context)
