from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel


class CaptionProcessorMixin(ABC, BaseModel):
    type: str

    @abstractmethod
    def process(self, caption: str) -> str:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.process(*args, **kwds)


class CaptionPassthrough(CaptionProcessorMixin):
    """
    Do nothing
    """

    type: Literal["passthrough"] = "passthrough"

    def process(self, caption: str) -> str:
        return caption
