from typing import Literal


from .util import CaptionProcessorMixin


class CaptionPrefix(CaptionProcessorMixin):
    type: Literal["prefix"] = "prefix"

    prefix: str

    def process(self, caption: str) -> str:
        return self.prefix + caption


class CaptionSuffix(CaptionProcessorMixin):
    type: Literal["suffix"] = "suffix"

    suffix: str

    def process(self, caption: str) -> str:
        return self.suffix + caption
