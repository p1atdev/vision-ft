from abc import ABC, abstractmethod
from typing import Literal

import torch
import torch.nn as nn

from .config import PeftConfigMixin


class PeftLayer(ABC, nn.Module):
    adapter_param_names: list[str]
    adapter_weight_names: list[str]

    enabled: bool

    @abstractmethod
    def init_weights(self) -> None:
        pass

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    @abstractmethod
    def from_weights(
        cls,
        adapter_weights: dict[str, torch.Tensor],
        original_layer: nn.Module,
    ) -> "PeftLayer":
        pass
