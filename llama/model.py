__all__ = ['Model']

from attrs import define, field


@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str


    def __str__(self):
        return f'{self.creator_hf_repo}:{self.hf_repo}:{self.hf_file}'
