from .util import CaptionProcessorMixin, CaptionPassthrough
from .shuffle import CaptionShuffle
from .prefix import CaptionPrefix

CaptionProcessorList = list[CaptionPassthrough | CaptionPrefix | CaptionShuffle]
