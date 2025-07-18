import torch
import torch.nn as nn

from typing import Literal

from .config import PeftConfigMixin
from .util import PeftLayer
from ...utils.dtype import str_to_dtype


class LoHaLiteConfig(PeftConfigMixin):
    type: Literal["loha_lite"] = "loha_lite"
    dropout: float = 0.0


class LoHaLiteLinear(PeftLayer):
    adapter_param_names = [
        "loha_up",
        "loha_down",
    ]
    adapter_weight_names = [
        "loha_up",
        "loha_down",
    ]

    def __init__(
        self,
        config: LoHaLiteConfig,
        original_linear: nn.Linear,
    ) -> None:
        super().__init__()

        self.config = config

        # get original parameters
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # loha lite modules
        self.loha_down = nn.Parameter(torch.empty(in_features, 1))
        self.loha_up = nn.Parameter(torch.empty(1, out_features))

        # loha_weight = W ⨀ loha_down ⨀ loha_up
        # new_weight = original_linear.weight + loha_weight

        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )

        # enable/disable LoHa
        self.enabled = True

        # original linear
        self.linear = original_linear
        # freeze original linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        self.init_weights()

    def init_weights(self) -> None:
        self.dropout.to_empty(device=self.linear.weight.device)

        # all ones
        nn.init.ones_(self.loha_down)
        nn.init.ones_(self.loha_up)

        self.dropout = (
            nn.Dropout(self.config.dropout)
            if self.config.dropout > 0
            else nn.Identity()
        )

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_output = self.linear(x)

        # if disabled, return original output
        if not self.enabled:
            return original_output

        # LoHa Lite
        loha_output = original_output * (self.loha_down * self.loha_up).to(
            original_output.dtype
        )

        return loha_output

    def train(self, mode: bool = True) -> "LoHaLiteLinear":
        # set parameters' training mode
        self.loha_down.requires_grad_(mode)
        self.loha_up.requires_grad_(mode)

        # do not train original linear layer
        self.linear.train(False)

        return self

    def requires_grad_(self, requires_grad: bool = True) -> "LoHaLiteLinear":
        self.loha_down.requires_grad_(requires_grad)
        self.loha_up.requires_grad_(requires_grad)

        # do not train original linear layer
        self.linear.weight.requires_grad_(False)

        return self

    @classmethod
    def from_weights(
        cls,
        adapter_weights: dict[str, torch.Tensor],
        original_layer: nn.Linear,
    ) -> "LoHaLiteLinear":
        config = LoHaLiteConfig()
        module = cls(
            config,
            original_layer,
        )
        module.loha_down = nn.Parameter(
            adapter_weights["loha_down"].to(original_layer.weight.device)
        )
        module.loha_up = nn.Parameter(
            adapter_weights["loha_up"].to(original_layer.weight.device)
        )

        return module
