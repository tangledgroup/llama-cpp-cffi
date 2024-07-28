__all__ = ['Model']

from typing import Optional

from attrs import define, field


@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str
    tokenizer_hf_repo: Optional[str] = None


    def __str__(self):
        if self.tokenizer_hf_repo:
            return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.tokenizer_hf_repo}'
        else:
            return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}'
