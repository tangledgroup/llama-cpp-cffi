__all__ = ['Model']

from attrs import define, field


@define
class Model:
    creator_hf_repo: str
    hf_repo: str
    hf_file: str
