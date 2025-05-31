from abc import ABC, abstractmethod
from PIL import Image
from pydantic import BaseModel

import torch


class RewardModelMixin(ABC):
    @abstractmethod
    def __call__(
        self,
        images: list[Image.Image],
        prompts: list[str],
        # TODO: more args?
    ) -> torch.Tensor:
        pass


class RewardModelConfig(ABC, BaseModel):
    type: str

    @abstractmethod
    def load_model(self, device: torch.device) -> RewardModelMixin:
        """
        Load the model on the specified device.
        This method should be implemented in subclasses to return an instance of RewardModelMixin.
        """
        raise NotImplementedError("Subclasses must implement this method.")
