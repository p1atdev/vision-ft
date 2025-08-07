from typing import Literal
import random


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
        return caption + self.suffix


class CaptionRandomPrefix(CaptionProcessorMixin):
    type: Literal["prefix_random"] = "prefix_random"

    prefix: list[str]

    def process(self, caption: str) -> str:
        prefix = random.choice(self.prefix)

        return prefix + caption


class CaptionRandomSuffix(CaptionProcessorMixin):
    type: Literal["suffix_random"] = "suffix_random"

    suffix: list[str]

    def process(self, caption: str) -> str:
        suffix = random.choice(self.suffix)

        return caption + suffix
