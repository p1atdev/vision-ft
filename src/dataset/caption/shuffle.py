from typing import Literal
import random

from .util import CaptionProcessorMixin


class CaptionShuffle(CaptionProcessorMixin):
    type: Literal["shuffle"] = "shuffle"

    split_separator: str = ","
    trim: bool = True

    concat_separator: str = ", "

    def process(self, caption: str) -> str:
        items = [
            item.strip() if self.trim else item
            for item in caption.split(self.split_separator)
        ]
        random.shuffle(items)

        return self.concat_separator.join(items)
