import torch
import torch.nn as nn

from typing import Literal

from .config import PeftConfigMixin
from .util import PeftLayer
from ...utils.dtype import str_to_dtype


class LoRAConfig(PeftConfigMixin):
    type: Literal["lora"] = "lora"
    rank: int
    alpha: float = 1.0
    dropout: float = 0.0
    use_bias: bool = False


class LoRALinear(PeftLayer):
    adapter_param_names = ["lora_up", "lora_down", "alpha"]
    adapter_weight_names = [
        "lora_up.weight",
        "lora_up.bias",
        "lora_down.weight",
        "alpha",
    ]

    def __init__(
        self,
        config: LoRAConfig,
        original_linear: nn.Linear,
    ) -> None:
        super().__init__()

        self.config = config
        dtype = str_to_dtype(config.dtype)

        # get original parameters
        in_features = original_linear.in_features
        out_features = original_linear.out_features

        # lora modules
        self.lora_down = nn.Linear(in_features, config.rank, bias=False, dtype=dtype)
        self.lora_up = nn.Linear(config.rank, out_features, bias=False, dtype=dtype)
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=dtype))
        self.rank = config.rank
        if config.use_bias:
            self.lora_up.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))

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

    @classmethod
    def from_weights(
        cls,
        adapter_weights: dict[str, torch.Tensor],
        original_layer: nn.Linear,
    ) -> "LoRALinear":
        rank = adapter_weights["lora_down.weight"].shape[0]
        alpha = adapter_weights["alpha"].item()

        config = LoRAConfig(rank=rank, alpha=alpha)
        module = cls(
            config,
            original_layer,
        )
        module.lora_down.weight = nn.Parameter(
            adapter_weights["lora_down.weight"].to(original_layer.weight.device)
        )
        module.lora_up.weight = nn.Parameter(
            adapter_weights["lora_up.weight"].to(original_layer.weight.device)
        )
        module.alpha = nn.Parameter(
            adapter_weights["alpha"].to(original_layer.weight.device)
        )
        if "bias" in adapter_weights:
            module.lora_up.bias = nn.Parameter(
                adapter_weights["lora_up.bias"].to(original_layer.weight.device)
            )

        return module


# ref: https://github.com/kohya-ss/sd-scripts/blob/52c8dec9534e9dea1226bf6e8d6ad3b1483d63aa/networks/lora.py#L59
class LoRAConv2d(PeftLayer):
    adapter_param_names = ["lora_up", "lora_down", "alpha"]
    adapter_weight_names = [
        "lora_up.weight",
        "lora_up.bias",
        "lora_down.weight",
        "alpha",
    ]

    def __init__(
        self,
        config: LoRAConfig,
        original_conv2d: nn.Conv2d,
    ) -> None:
        super().__init__()

        self.config = config
        dtype = str_to_dtype(config.dtype)

        # get original parameters
        in_channels = original_conv2d.in_channels
        out_channels = original_conv2d.out_channels
        kernel_size = original_conv2d.kernel_size
        stride = original_conv2d.stride
        padding = original_conv2d.padding

        # lora modules
        self.lora_down = nn.Conv2d(
            in_channels,
            config.rank,
            kernel_size=kernel_size,  # type: ignore
            stride=stride,  # type: ignore
            padding=padding,  # type: ignore
            bias=False,
            dtype=dtype,
        )
        self.lora_up = nn.Conv2d(
            config.rank,
            out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
            dtype=dtype,
        )
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.alpha = nn.Parameter(torch.tensor(config.alpha, dtype=dtype))
        self.rank = config.rank
        if config.use_bias:
            self.lora_up.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype))

        # enable/disable LoRA
        self.enabled = True

        # original convolution
        self.conv2d = original_conv2d
        # freeze original convolution
        self.conv2d.weight.requires_grad_(False)
        if self.conv2d.bias is not None:
            self.conv2d.bias.requires_grad_(False)

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
        original_output = self.conv2d(x)

        # if disabled, return original output
        if not self.enabled:
            return original_output

        # LoRA
        lora_down = self.lora_down(self.dropout(x))
        lora_up = self.lora_up(lora_down)
        lora_output = lora_up * (self.alpha / self.rank)

        return original_output + lora_output

    @classmethod
    def from_weights(
        cls,
        adapter_weights: dict[str, torch.Tensor],
        original_layer: nn.Conv2d,
    ) -> "LoRAConv2d":
        rank = adapter_weights["lora_down.weight"].shape[0]
        alpha = adapter_weights["alpha"].item()

        config = LoRAConfig(rank=rank, alpha=alpha)
        module = cls(
            config,
            original_layer,
        )
        module.lora_down.weight = nn.Parameter(
            adapter_weights["lora_down.weight"].to(original_layer.weight.device)
        )
        module.lora_up.weight = nn.Parameter(
            adapter_weights["lora_up.weight"].to(original_layer.weight.device)
        )
        module.alpha = nn.Parameter(
            adapter_weights["alpha"].to(original_layer.weight.device)
        )
        if "bias" in adapter_weights:
            module.lora_up.bias = nn.Parameter(
                adapter_weights["lora_up.bias"].to(original_layer.weight.device)
            )

        return module
