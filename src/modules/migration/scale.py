import torch
import torch.nn as nn


class MigrationScaleFromZero(nn.Module):
    def __init__(self, dim: int = 1):
        super().__init__()

        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(dim))

    def scale_positive(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.scale

    def scale_negative(self, input: torch.Tensor) -> torch.Tensor:
        return input * (1 - self.scale)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # This module would be never loaded from state_dict,
        # and this function is called only before training.
        # So, we can initialize the scale parameter with zeros.

        self.scale = nn.Parameter(torch.zeros(self.dim))
