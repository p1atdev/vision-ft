from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle
from .append import CaptionPrefix, CaptionSuffix
from .drop import CaptionDrop, CaptionTagDrop

CaptionProcessorList = list[
    CaptionPassthrough
    | CaptionPrefix
    | CaptionSuffix
    | CaptionShuffle
    | CaptionDrop
    | CaptionTagDrop
]
