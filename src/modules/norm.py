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
