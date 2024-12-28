import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LoRAConfig


class LoRALinear(nn.Linear):
    def __init__(
        self,
        config: LoRAConfig,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dropout: float = 0.0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

        self.config = config

        # lora modules
        self.lora_down = nn.Linear(in_features, config.rank, bias=False)
        self.lora_up = nn.Linear(config.rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.alpha = nn.Parameter(torch.tensor(config.alpha))
        self.rank = config.rank

        # enable/disable LoRA
        self.enabled = True

        # freeze original weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        # freeze LoRA alpha
        self.alpha.requires_grad = False

        self.init_weights()

    def init_weights(self) -> None:
        # following: https://github.com/pytorch/torchtune/blob/aa8f365f91a69aa36aaea14cf6f03ccd45310bb6/torchtune/modules/peft/lora.py#L102-L106
        nn.init.kaiming_uniform_(self.lora_down.weight)
        nn.init.zeros_(self.lora_up.weight)

    @property
    def adapter_param_names(self) -> list[str]:
        return ["lora_up.weight", "lora_down.weight", "alpha"]

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = F.linear(x, self.weight, self.bias)

        # if disabled, return original output
        if not self.enabled:
            return original_output

        # LoRA
        lora_down = self.lora_down(self.dropout(x))
        lora_up = self.lora_up(lora_down)
        lora_output = lora_up * (self.alpha / self.rank)

        return original_output + lora_output
