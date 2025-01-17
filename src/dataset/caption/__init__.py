from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle
from .prefix import CaptionPrefix
from .drop import CaptionDrop

CaptionProcessorList = list[
    CaptionPassthrough | CaptionPrefix | CaptionShuffle | CaptionDrop
]
