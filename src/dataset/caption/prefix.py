from typing import Literal


from .util import CaptionProcessorMixin


class CaptionPrefix(CaptionProcessorMixin):
    type: Literal["prefix"] = "prefix"

    prefix: str

    def process(self, caption: str) -> str:
        return self.prefix + caption
