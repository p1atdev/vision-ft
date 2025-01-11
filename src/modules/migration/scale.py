import torch
import torch.nn as nn


class MigrationScaleFromZero(nn.Module):
    """
    Start from 0 and gradually increase to 1.
    """

    def __init__(
        self,
        dim: int = 1,
        freezing_threshold: float | None = None,
    ):
        super().__init__()

        self.dim = dim
        self.scale = nn.Parameter(torch.zeros(dim))
        self.freezing_threshold = freezing_threshold

    @property
    def inner_scale(self) -> torch.Tensor:
        if (threshold := self.freezing_threshold) is not None and (
            1 - self.scale
        ).abs().max() < threshold:
            return torch.ones_like(self.scale.detach())

        return self.scale

    def scale_positive(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return input * self.inner_scale

    def scale_negative(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        return input * (1 - self.inner_scale)

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
