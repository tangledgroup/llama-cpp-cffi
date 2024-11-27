__all__ = ['Model']

from typing import Optional

from attrs import define


@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str
    mmproj_hf_file: Optional[str] = ''
    tokenizer_hf_repo: Optional[str] = ''


    def __str__(self):
        # if self.tokenizer_hf_repo:
        #     return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.tokenizer_hf_repo}'
        # else:
        #     return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}'
        return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}:{self.mmproj_hf_file}:{self.tokenizer_hf_repo}'
