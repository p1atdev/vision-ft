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
