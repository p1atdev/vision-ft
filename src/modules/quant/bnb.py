from typing import Literal

import torch
import torch.nn as nn

import bitsandbytes as bnb


def collect_quantized_stats(
    prefix: str,
    state_dict: dict[str, torch.Tensor],
    remove_prefix: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Collect keys that start with prefix
    """
    stats = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stats[key[len(prefix) :] if remove_prefix else key] = value

    return stats


def _get_bnb_4bit_quant_type_from_stats(
    quantized_stats: dict[str, torch.Tensor],
) -> Literal["fp4", "nf4"]:
    for key in quantized_stats.keys():
        if "quant_state" in key:
            quant_type = key[len("quant_state.bitsandbytes__") :]
            assert quant_type == "nf4" or quant_type == "fp4"
            return quant_type

    raise ValueError("quant_type not found")


class BnbLinear4bit(bnb.nn.Linear4bit):
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
            device=device,
        )

        self.weight = bnb.nn.Params4bit(
            torch.empty(
                output_features,
                input_features,
                dtype=compute_dtype,
                device="meta",
            ),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(output_features, dtype=compute_dtype, device="meta"),
                requires_grad=False,
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
        quantized_stats = collect_quantized_stats(
            f"{prefix}weight.",
            state_dict,
        )

        if len(quantized_stats) > 0:
            # load prequantized weights
            quant_type = _get_bnb_4bit_quant_type_from_stats(quantized_stats)
            weight = bnb.nn.Params4bit.from_prequantized(
                data=state_dict[f"{prefix}weight"],
                quantized_stats=quantized_stats,
                quant_type=quant_type,
                module=self,
            )
            setattr(self, "weight", weight)

            if self.bias is not None:
                bias = nn.Parameter(
                    state_dict[f"{prefix}bias"],
                    requires_grad=False,
                )
                setattr(self, "bias", bias)
        else:
            # load from full precision weights
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )

            # nn.Module uses nn.Parameters directly, so we
            # need to convert it to bnb.nn.Params4bit manually
            weight = bnb.nn.Params4bit(
                self.weight.data,
                requires_grad=False,
            )
            weight.compress_statistics = self.compress_statistics
            weight.quant_type = self.quant_type
            weight.quant_storage = self.quant_storage
            setattr(self, "weight", weight)


class BnbLinear8bit(bnb.nn.Linear8bitLt):
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
