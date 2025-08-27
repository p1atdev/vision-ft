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


class CaptionShuffleInGroup(CaptionProcessorMixin):
    type: Literal["shuffle_in_group"] = "shuffle_in_group"

    group_separator: str = "|||"
    split_separator: str = ","
    trim: bool = True

    concat_separator: str = ", "

    def shuffle(self, group: str) -> str:
        items = [
            item.strip() if self.trim else item
            for item in group.split(self.split_separator)
        ]
        random.shuffle(items)

        return self.concat_separator.join(items)

    def process(self, caption: str) -> str:
        groups = caption.split(self.group_separator)
        shuffled_groups = [self.shuffle(group) for group in groups]
        return self.concat_separator.join(shuffled_groups)
