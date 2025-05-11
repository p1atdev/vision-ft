from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle, CaptionShuffleInGroup
from .append import CaptionPrefix, CaptionSuffix
from .drop import CaptionDrop, CaptionTagDrop

CaptionProcessorList = list[
    CaptionPassthrough
    | CaptionPrefix
    | CaptionSuffix
    | CaptionShuffle
    | CaptionShuffleInGroup
    | CaptionDrop
    | CaptionTagDrop
]
