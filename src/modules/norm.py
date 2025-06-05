from typing import NamedTuple


import torch
import torch.nn as nn
import torch.nn.functional as F


class FP32LayerNorm(nn.LayerNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            hidden_states.to(torch.float32),
            self.normalized_shape,
            self.weight.to(torch.float32) if self.weight is not None else None,
            self.bias.to(torch.float32) if self.bias is not None else None,
            self.eps,
        ).to(hidden_states.dtype)


class SingleAdaLayerNormZeroOutput(NamedTuple):
    hidden_states: torch.Tensor
    scale: torch.Tensor
    shift: torch.Tensor
    gate: torch.Tensor


class SingleAdaLayerNormZero(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()

        self.act = nn.SiLU()
        self.norm = FP32LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            embedding_dim,
            3 * hidden_dim,  # 3 for scale, shift, gate
            bias=True,
        )

    def init_weights(self) -> None:
        self.linear.to_empty(device=torch.device("cpu"))

        # init with zeros!
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        time_embed: torch.Tensor,
    ) -> SingleAdaLayerNormZeroOutput:
        norm_hidden_states = self.norm(hidden_states)

        embedding = self.linear(self.act(time_embed))
        scale, shift, gate = embedding.chunk(3, dim=1)

        hidden_states = norm_hidden_states * (1 + scale.unsqueeze(1)) + shift.unsqueeze(
            1
        )

        return SingleAdaLayerNormZeroOutput(
            hidden_states=hidden_states,
            scale=scale,
            shift=shift,
            gate=gate,  # will be used later
        )
