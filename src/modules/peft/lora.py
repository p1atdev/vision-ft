import torch
import torch.nn as nn

from .config import LoRAConfig


class LoRALinear(nn.Module):
    adapter_param_names = ["lora_up", "lora_down", "alpha"]

    def __init__(
        self,
        config: LoRAConfig,
        original_linear: nn.Linear,
        in_features: int,
        out_features: int,
        dropout: float = 0.0,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.config = config

        # lora modules
        self.lora_down = nn.Linear(in_features, config.rank, bias=False, dtype=dtype)
        self.lora_up = nn.Linear(config.rank, out_features, bias=False, dtype=dtype)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=dtype))
        self.rank = config.rank

        # enable/disable LoRA
        self.enabled = True

        # original linear
        self.linear = original_linear
        # freeze original linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        # freeze LoRA alpha
        self.alpha.requires_grad_(False)

        self.init_weights()

    def init_weights(self) -> None:
        # following: https://github.com/pytorch/torchtune/blob/aa8f365f91a69aa36aaea14cf6f03ccd45310bb6/torchtune/modules/peft/lora.py#L102-L106
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

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
