import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from .config import PeftConfigMixin
from .util import PeftLayer
from ...utils.dtype import str_to_dtype

# https://github.com/KohakuBlueleaf/LyCORIS/blob/main/lycoris/modules/loha.py
# https://ai-joyful.com/lycoris/#toc7


class LoHaConfig(PeftConfigMixin):
    type: Literal["loha"] = "loha"
    rank: int
    alpha: float = 1.0
    dropout: float = 0.0


class LoHaLinear(PeftLayer):
    adapter_param_names = [
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "alpha",
    ]
    adapter_weight_names = [
        "hada_w1_a",
        "hada_w1_b",
        "hada_w2_a",
        "hada_w2_b",
        "alpha",
    ]

    def __init__(
        self,
        config: LoHaConfig,
        original_linear: nn.Linear,
    ) -> None:
        super().__init__()

        self.config = config
        dtype = str_to_dtype(config.dtype)

        # get original parameters
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # loha modules
        self.hada_w1_a = nn.Parameter(torch.empty(in_features, config.rank))
        self.hada_w1_b = nn.Parameter(torch.empty(config.rank, out_features))
        # w1 = w1_a @ w1_b

        self.hada_w2_a = nn.Parameter(torch.empty(in_features, config.rank))
        self.hada_w2_b = nn.Parameter(torch.empty(config.rank, out_features))
        # w2 = w2_a @ w2_b

        # loha_weight = w1 ⨀ w2
        # new_weight = original_linear.weight + loha_weight

        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.alpha = nn.Parameter(
            torch.tensor(config.alpha, dtype=dtype),
            requires_grad=False,
        )
        self.rank = config.rank

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

        nn.init.normal_(self.hada_w1_b, std=1)
        nn.init.normal_(self.hada_w1_a, std=0.1)
        nn.init.normal_(self.hada_w2_b, std=1)
        nn.init.constant_(self.hada_w2_a, 0)  # last is zero

        self.alpha = nn.Parameter(
            torch.tensor(self.config.alpha, dtype=self.hada_w1_a.dtype),
            requires_grad=False,
        )
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

        # LoHa
        loha_down_weight = self.hada_w1_a @ self.hada_w1_b
        loha_up_weight = self.hada_w2_a @ self.hada_w2_b

        # hadmard production
        loha_weight = (loha_down_weight * loha_up_weight).T
        print(f"LoHa weight shape: {loha_weight.shape}")
        loha_output = F.linear(x, loha_weight.to(x.dtype)) * (self.alpha / self.rank)

        return original_output + loha_output

    def train(self, mode: bool = True) -> "LoHaLinear":
        # set parameters' training mode
        self.hada_w1_a.requires_grad_(mode)
        self.hada_w1_b.requires_grad_(mode)
        self.hada_w2_a.requires_grad_(mode)
        self.hada_w2_b.requires_grad_(mode)

        # do not train original linear layer
        self.linear.train(False)

        return self

    def requires_grad_(self, requires_grad: bool = True) -> "LoHaLinear":
        self.hada_w1_a.requires_grad_(requires_grad)
        self.hada_w1_b.requires_grad_(requires_grad)
        self.hada_w2_a.requires_grad_(requires_grad)
        self.hada_w2_b.requires_grad_(requires_grad)

        # do not train original linear layer
        self.linear.weight.requires_grad_(False)

        return self

    @classmethod
    def from_weights(
        cls,
        adapter_weights: dict[str, torch.Tensor],
        original_layer: nn.Linear,
    ) -> "LoHaLinear":
        rank = adapter_weights["hada_w1_a"].shape[1]
        alpha = adapter_weights["alpha"].item()

        config = LoHaConfig(rank=rank, alpha=alpha)
        module = cls(
            config,
            original_layer,
        )
        module.hada_w1_a = nn.Parameter(
            adapter_weights["hada_w1_a"].to(original_layer.weight.device)
        )
        module.hada_w1_b = nn.Parameter(
            adapter_weights["hada_w1_b"].to(original_layer.weight.device)
        )
        module.hada_w2_a = nn.Parameter(
            adapter_weights["hada_w2_a"].to(original_layer.weight.device)
        )
        module.hada_w2_b = nn.Parameter(
            adapter_weights["hada_w2_b"].to(original_layer.weight.device)
        )

        module.alpha = nn.Parameter(
            adapter_weights["alpha"].to(original_layer.weight.device)
        )

        return module

    def load_weights(
        self,
        adapter_weights: dict[str, torch.Tensor | None],
    ) -> None:
        device = self.hada_w1_a.device

        if (weight := adapter_weights.get("hada_w1_a")) is not None:
            self.hada_w1_a = nn.Parameter(weight.to(device))
        if (weight := adapter_weights.get("hada_w1_b")) is not None:
            self.hada_w1_b = nn.Parameter(weight.to(device))
        if (weight := adapter_weights.get("hada_w2_a")) is not None:
            self.hada_w2_a = nn.Parameter(weight.to(device))
        if (weight := adapter_weights.get("hada_w2_b")) is not None:
            self.hada_w2_b = nn.Parameter(weight.to(device))
        if (weight := adapter_weights.get("alpha")) is not None:
            self.alpha = nn.Parameter(weight.to(device))
