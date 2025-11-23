from typing import Literal


from .util import CaptionProcessorMixin


class CaptionReplace(CaptionProcessorMixin):
    type: Literal["replace"] = "replace"

    source: str
    target: str

    def process(self, caption: str) -> str:
        return caption.replace(self.source, self.target)
