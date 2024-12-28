import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb

from modules.peft.config import QLoRAConfig, QUANT_TYPE


def get_linear_cls(quant_type: QUANT_TYPE):
    if quant_type == "fp8_e4m3fn":
        return nn.Linear

    if quant_type == "int8":
        return bnb.nn.Linear8bitLt

    if quant_type == "fp4":
        return bnb.nn.LinearFP4

    if quant_type == "nf4":
        return bnb.nn.LinearNF4

    raise ValueError(f"Unknown quant_type: {quant_type}")


class QLoRALinear(nn.Module):
    def __init__(
        self,
        config: QLoRAConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config
        self.quant_type = config.quant_type

        # lora modules
        self.lora_down = nn.Linear(in_features, config.rank, bias=False)
        self.lora_up = nn.Linear(config.rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(config.alpha))
        self.rank = config.rank

        # enable/disable LoRA
        self.enabled = True

        # set original linear
        self.linear = get_linear_cls(self.quant_type)(
            in_features,
            out_features,
            bias=bias,
        )
        # freeze original weights
        self.linear.weight.requires_grad = False

        # freeze LoRA alpha
        self.alpha.requires_grad = False

        self.init_weights()

    def init_weights(self) -> None:
        # following: https://github.com/pytorch/torchtune/blob/aa8f365f91a69aa36aaea14cf6f03ccd45310bb6/torchtune/modules/peft/lora.py#L102-L106
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    @property
    def adapter_param_names(self) -> list[str]:
        return ["lora_up.weight", "lora_down.weight", "alpha"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.linear(x)

        # if disabled, return original output
        if not self.enabled:
            return original_output

        # LoRA
        lora_down = self.lora_down(self.dropout(x))
        lora_up = self.lora_up(lora_down)
        lora_output = lora_up * (self.alpha / self.rank)

        return original_output + lora_output
