from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle
from .prefix import CaptionPrefix
from .drop import CaptionDrop, CaptionTagDrop

CaptionProcessorList = list[
    CaptionPassthrough | CaptionPrefix | CaptionShuffle | CaptionDrop | CaptionTagDrop
]
