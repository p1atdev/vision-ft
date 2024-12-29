import torch
import torch.nn as nn

import torchao as ao
from torchao.float8 import float8_linear as ao_fp8


class AOLinearNF4(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
        block_size: int = 64,
        scaler_block_size: int = 256,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
        )

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        nf4_weight = ao.dtypes.nf4tensor.to_nf4(
            self.weight,
            block_size=block_size,
            scaler_block_size=scaler_block_size,
        )
        torch.utils.swap_tensors(
            self.weight, nn.Parameter(nf4_weight, requires_grad=False)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = ao.dtypes.nf4tensor.linear_nf4(
            input,
            self.weight,
        )
        if self.bias is not None:
            out += self.bias

        return out

    @classmethod
    @torch.no_grad()
    def from_module(
        cls, module: nn.Module, block_size: int = 64, scaler_block_size: int = 256
    ):
        assert isinstance(module, nn.Linear)
        new_linear = cls(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            block_size=block_size,
            scaler_block_size=scaler_block_size,
        )

        return new_linear


class AOLinearFP8(ao_fp8.Float8Linear):
    pass
