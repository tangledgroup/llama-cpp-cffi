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
    text_completions,
    clip_completions,
    mllama_completions,
)


@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str
    mmproj_hf_file: Optional[str] = ''
    tokenizer_hf_repo: Optional[str] = ''
    _model: Any = None
    _options: Options = Options()

    def __str__(self):
        return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.mmproj_hf_file}:{self.tokenizer_hf_repo}'


    def init(self, **options: Unpack[Options]):
        # print(f'! {options=}')
        options = Options(
            **options,
            model=self,
        )

        self._model = model_init(options)
        self._options = options


    def completions(self, **options: Unpack[Options]) -> Iterator[str]:
        options = Options(
            **(asdict(self._options, recurse=False) | options),
        )

        _context = context_init(self._model, options)
        _sampler = sampler_init(options)

        for token in text_completions(self._model, _context, _sampler, options):
            yield token

        sampler_free(_sampler)
        context_free(_context)
