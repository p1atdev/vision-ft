from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

import bitsandbytes as bnb
from bitsandbytes.functional import (
    quantize_4bit,
    int8_double_quant,
)

from src.utils.tensor import is_target_key

QUANT_TYPE = Literal[
    "fp8_e4m3fn",
    "int8",
    "fp4",
    "nf4",
]


def _get_quantized_linear(
    module: nn.Module,
    quant_type: QUANT_TYPE,
) -> nn.Module:
    if quant_type == "nf4" or quant_type == "fp4":
        return Linear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type=quant_type,
        )
    elif quant_type == "int8":
        return Linear8bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
        )
    elif quant_type == "fp8_e4m3fn":
        return module  # as is
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def _replace_quantized_linear(
    module: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
    prefix: str = "",
) -> None:
    for name, layer in module.named_children():
        full_name = f"{prefix}{name}"

        if isinstance(layer, nn.Linear):
            if not is_target_key(full_name, include_keys, exclude_keys):
                continue
            # replace with quantized module
            quantized_module = _get_quantized_linear(layer, quant_type)
            quantized_module.requires_grad_(False)
            setattr(module, name, quantized_module)
        else:
            _replace_quantized_linear(
                layer,
                quant_type,
                include_keys,
                exclude_keys,
                f"{full_name}.",
            )


# just wrap
def replace_quantized_linear(
    model: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> nn.Module:
    """
    Replace the Linear layer with quantized Linear layer
    """
    _replace_quantized_linear(
        model,
        quant_type,
        include_keys,
        exclude_keys,
    )
    return model


class Linear4bit(bnb.nn.Linear4bit):
    def __init__(
        self,
        input_features,
        output_features,
        bias=True,
        compute_dtype=None,
        compress_statistics=True,
        quant_type="fp4",
        quant_storage=torch.uint8,
        device=None,
    ):
        super().__init__(
            input_features,
            output_features,
            bias=bias,
            compute_dtype=compute_dtype,
            compress_statistics=compress_statistics,
            quant_type=quant_type,
            quant_storage=quant_storage,
            device=device,
        )

        self.compute_dtype = compute_dtype
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_storage = quant_storage

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # if assign=True, nn.Module uses nn.Parameters directly.
        # so we need to convert it to bnb.nn.Params4bit manually
        if "assign_to_params_buffers" in local_metadata:
            self.weight = bnb.nn.Params4bit(
                self.weight.data,
                requires_grad=False,
            )
            setattr(self, "weight", self.weight)


class Linear8bit(bnb.nn.Linear8bitLt):
    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

        # if assign=True, nn.Module uses nn.Parameters directly.
        # so we need to convert it to bnb.nn.Int8Params manually
        if "assign_to_params_buffers" in local_metadata:
            self.weight = bnb.nn.Int8Params(
                self.weight.data,
                requires_grad=False,
            )
            setattr(self, "weight", self.weight)
