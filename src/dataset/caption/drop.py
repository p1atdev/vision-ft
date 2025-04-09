from typing import Literal
import random


from .util import CaptionProcessorMixin


class CaptionDrop(CaptionProcessorMixin):
    type: Literal["drop"] = "drop"

    drop_rate: float

    def process(self, caption: str) -> str:
        if random.random() < self.drop_rate:
            return ""

        return caption


class CaptionTagDrop(CaptionProcessorMixin):
    type: Literal["tag_drop"] = "tag_drop"

    drop_rate: float
    separator: str = ","

    def process(self, caption: str) -> str:
        tags = caption.split(self.separator)

        tags = [tag for tag in tags if random.random() >= self.drop_rate]
        return self.separator.join(tags)
