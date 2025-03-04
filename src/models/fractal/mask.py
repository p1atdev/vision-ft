from abc import ABC

import torch
import torch.nn as nn


class MaskGenerator(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UniformMaskGenerator(MaskGenerator):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, patches: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _hidden_dim = patches.shape

        num_masked_tokens = torch.randint(
            1, seq_len + 1, (batch_size,), device=patches.device
        )
        mask = torch.zeros(batch_size, seq_len, device=patches.device)
        mask = mask.scatter(dim=1, index=orders[:, :num_masked_tokens], value=1)

        return mask


class TruncatedNormalMaskGenerator(MaskGenerator):
    def __init__(self, std: float = 0.25) -> None:
        super().__init__()

        self.std = std

    def forward(self, patches: torch.Tensor, orders: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _hidden_dim = patches.shape

        mask_rates = torch.zeros(batch_size, device=patches.device)
        # generate randomly from [0, 1]
        mask_rates = nn.init.trunc_normal_(
            mask_rates, mean=1.0, std=self.std, a=0.0, b=1.0
        )
        num_masked_tokens = torch.ceil(mask_rates * seq_len)

        indices = torch.arange(seq_len, device=patches.device).expand(
            batch_size, seq_len
        )
        orders_position = torch.argsort(orders, dim=-1)

        mask = indices < num_masked_tokens.unsqueeze(-1)
        mask = torch.scatter(
            torch.zeros_like(mask, device=patches.device),
            dim=-1,
            index=orders_position,
            src=mask,
        )

        return mask
