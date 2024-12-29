from typing import Literal, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchao as ao
from torchao.float8 import float8_linear as ao_fp8
import bitsandbytes as bnb
from bitsandbytes.functional import quantize_4bit, int8_double_quant
import optimum.quanto as quanto

from src.utils.tensor import is_target_key

QUANT_TYPE = Literal[
    "native_fp8_e4m3fn",
    "bnb_int8",
    "bnb_fp4",
    "bnb_nf4",
    "quanto_int4",
    "quanto_int8",
    "ao_nf4",
    "ao_fp8",
]


def _get_quantized_linear(
    module: nn.Module,
    quant_type: QUANT_TYPE,
) -> nn.Module:
    ## Bitsandbytes
    if quant_type == "bnb_nf4" or quant_type == "bnb_fp4":
        return BnbLinear4bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            quant_type=quant_type[len("bnb_") :],
        )
    elif quant_type == "bnb_int8":
        return BnbLinear8bit(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            has_fp16_weights=False,
        )

    ## TorchAO
    elif quant_type == "ao_nf4":
        return AOLinearNF4.from_module(module)

    elif quant_type == "ao_fp8":
        return AOLinearFP8.from_float(module)

    ## Native
    elif quant_type == "native_fp8_e4m3fn":
        return module.to(torch.float8_e4m3fn)  # as is

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
            del layer
        else:
            _replace_quantized_linear(
                layer,
                quant_type,
                include_keys,
                exclude_keys,
                f"{full_name}.",
            )


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


def _quantize_inplace(
    module: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> None:
    for name, layer in module.named_children():
        if isinstance(layer, nn.Linear):
            if not is_target_key(name, include_keys, exclude_keys):
                continue

            ## Bitsandbytes
            if quant_type == "bnb_nf4" or quant_type == "bnb_fp4":
                qlinear = BnbLinear4bit(
                    layer.in_features,
                    layer.out_features,
                    bias=layer.bias is not None,
                    quant_type=quant_type[len("bnb_") :],
                )
                assert isinstance(qlinear, BnbLinear4bit)
                qlinear.load_state_dict(layer.state_dict(), assign=True)

                setattr(module, name, qlinear)
                del layer

            elif quant_type == "bnb_int8":
                qlinear = BnbLinear8bit(
                    layer.in_features,
                    layer.out_features,
                    bias=layer.bias is not None,
                )
                assert isinstance(qlinear, BnbLinear8bit)
                qlinear.load_state_dict(layer.state_dict(), assign=True)

                setattr(module, name, qlinear)
                del layer

            ## Optimum Quanto
            elif quant_type == "quanto_int4":
                qlinear = QuantoLinear.from_module(
                    layer,
                    weights=quanto.qint4,
                )
                assert isinstance(qlinear, QuantoLinear)
                qlinear.freeze()
                setattr(module, name, qlinear)
                del layer

            elif quant_type == "quanto_int8":
                qlinear = QuantoLinear.from_module(
                    layer,
                    weights=quanto.qint8,
                )
                assert isinstance(qlinear, QuantoLinear)
                qlinear.freeze()
                setattr(module, name, qlinear)
                del layer
        else:
            _quantize_inplace(
                layer,
                quant_type,
                include_keys,
                exclude_keys,
            )


def quantize_inplace(
    model: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> None:
    _quantize_inplace(
        model,
        quant_type,
        include_keys,
        exclude_keys,
    )


def freeze_quantized_linear(model: nn.Module) -> None:
    # freeze quanto layer
    quanto.freeze(model)


def collect_children_keys(
    prefix: str,
    state_dict_keys: Iterable[str],
    remove_prefix: bool = True,
) -> list[str]:
    """
    Collect keys that start with prefix
    """
    keys = []
    for key in state_dict_keys:
        if key.startswith(prefix):
            keys.append(key[len(prefix) :] if remove_prefix else key)

    return keys


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


def get_quant_type_from_children_keys(
    children_keys: Iterable[str],
) -> QUANT_TYPE:
    for key in children_keys:
        if "quant_state" in key:
            quant_type = key[len("quant_state.bitsandbytes__") :]
            if quant_type in "nf4":
                return "bnb_nf4"
            elif quant_type in "fp4":
                return "bnb_fp4"
        elif "weight_format" in key:
            return "bnb_int8"

    raise ValueError("quant_type not found")


def _replace_with_prequantized_layers(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
) -> None:
    for name, layer in model.named_children():
        full_name = f"{prefix}{name}"
        if isinstance(layer, nn.Linear):
            children_keys = collect_children_keys(
                f"{full_name}.weight.",
                state_dict.keys(),
            )
            quant_type = get_quant_type_from_children_keys(children_keys)
            quantized_module = _get_quantized_linear(layer, quant_type)

            quantized_module.requires_grad_(False)
            setattr(model, name, quantized_module)
            del layer
        else:
            quant_type = _replace_with_prequantized_layers(
                layer,
                state_dict,
                f"{full_name}.",
            )


def replace_with_prequantized_layers(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    _replace_with_prequantized_layers(
        model,
        state_dict,
    )


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> dict[str, torch.Tensor]:
    if not (quant_type == "bnb_nf4" or quant_type == "bnb_fp4"):
        raise NotImplementedError("Only bitsandbytes 4bit quantization is supported")

    for key in list(state_dict.keys()):
        if not is_target_key(key, include_keys, exclude_keys):
            continue

        if quant_type == "bnb_nf4" or quant_type == "bnb_fp4":
            tensor, state = quantize_4bit(
                state_dict[key].cuda(),
                quant_type=quant_type[len("bnb_") :],  # nf4 or fp4
            )
            state_dict[key] = tensor.cpu()
            for state_key, state_value in state.as_dict(packed=True).items():
                state_dict[f"{key}.{state_key}"] = state_value
    return state_dict


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


class QuantoLinear(quanto.nn.QLinear):
    pass


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
