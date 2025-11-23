from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle, CaptionShuffleInGroup
from .append import (
    CaptionPrefix,
    CaptionSuffix,
    CaptionRandomPrefix,
    CaptionRandomSuffix,
)
from .drop import CaptionDrop, CaptionTagDrop
from .replace import CaptionReplace

CaptionProcessorList = list[
    CaptionPassthrough
    | CaptionPrefix
    | CaptionSuffix
    | CaptionRandomPrefix
    | CaptionRandomSuffix
    | CaptionShuffle
    | CaptionShuffleInGroup
    | CaptionDrop
    | CaptionTagDrop
    | CaptionReplace
]
