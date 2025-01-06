import torch
import torch.nn as nn


class MigrationScaleFromZero(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        self.scale = nn.Parameter(torch.zeros(dim))

    def scale_positive(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.scale

    def scale_negative(self, input: torch.Tensor) -> torch.Tensor:
        return input * (1 - self.scale)
