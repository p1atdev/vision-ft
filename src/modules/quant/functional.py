from typing import Literal, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import optimum.quanto as quanto
from bitsandbytes.functional import quantize_4bit

from ...utils.state_dict import get_target_keys

from .bnb import BnbLinear8bit, BnbLinear4bit
from .ao import AOLinearNF4, AOLinearFP8
from .quanto import QuantoLinear


QUANT_TYPE = Literal[
    "fp8_e4m3fn",
    "bnb_int8",
    "bnb_fp4",
    "bnb_nf4",
    "quanto_int4",
    "quanto_int8",
    "ao_nf4",
    "ao_fp8",
]


def validate_quant_type(quant_type: str) -> None:
    if quant_type not in [
        "fp8_e4m3fn",
        "bnb_int8",
        "bnb_fp4",
        "bnb_nf4",
        "quanto_int4",
        "quanto_int8",
        "ao_nf4",
        "ao_fp8",
    ]:
        raise ValueError(f"Unknown quant_type: {quant_type}")


def _get_quant_linear(
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
    elif quant_type == "fp8_e4m3fn":
        return module.to(torch.float8_e4m3fn)  # as is

    ## Optimum Quanto
    elif quant_type == "quanto_int4" or quant_type == "quanto_int8":
        raise NotImplementedError("QuantoLinear is not supported")

    raise ValueError(f"Unknown quant_type: {quant_type}")


def _replace_to_quant_linear(
    module: nn.Module,
    quant_type: QUANT_TYPE,
    target_keys: list[str],
    prefix: str = "",
) -> None:
    for name, layer in module.named_children():
        full_name = f"{prefix}{name}"

        if isinstance(layer, nn.Linear):
            if full_name not in target_keys:
                continue

            # replace with quantized module
            quantized_module = _get_quant_linear(layer, quant_type)
            quantized_module.requires_grad_(False)
            setattr(module, name, quantized_module)
            del layer
        else:
            _replace_to_quant_linear(
                layer,
                quant_type,
                target_keys,
                f"{full_name}.",
            )


def replace_to_quant_linear(
    model: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> nn.Module:
    """
    Replace the Linear layer with quantized Linear layer
    """
    target_keys = get_target_keys(
        include_keys,
        exclude_keys,
        [name for name, _ in model.named_modules()],
    )
    print(target_keys)
    _replace_to_quant_linear(
        model,
        quant_type,
        target_keys,
    )
    return model


def _quantize_inplace(
    module: nn.Module,
    quant_type: QUANT_TYPE,
    target_keys: list[str],
    prefix: str = "",
) -> None:
    for name, layer in module.named_children():
        full_name = f"{prefix}{name}"
        if isinstance(layer, nn.Linear):
            if full_name not in target_keys:
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

            ## Native
            elif quant_type == "fp8_e4m3fn":
                layer.to(torch.float8_e4m3fn)
        else:
            _quantize_inplace(
                layer,
                quant_type,
                target_keys,
                f"{full_name}.",
            )


def quantize_inplace(
    model: nn.Module,
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> None:
    target_keys = get_target_keys(
        include_keys,
        exclude_keys,
        [name for name, _ in model.named_modules()],
    )
    _quantize_inplace(
        model,
        quant_type,
        target_keys,
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


def _replace_by_prequantized_weights(
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
            if len(children_keys) == 0:
                continue

            quant_type = get_quant_type_from_children_keys(children_keys)
            quantized_module = _get_quant_linear(layer, quant_type)

            quantized_module.requires_grad_(False)
            setattr(model, name, quantized_module)
            del layer
        else:
            quant_type = _replace_by_prequantized_weights(
                layer,
                state_dict,
                f"{full_name}.",
            )


def replace_by_prequantized_weights(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    _replace_by_prequantized_weights(
        model,
        state_dict,
    )


def quantize_state_dict(
    state_dict: dict[str, torch.Tensor],
    quant_type: QUANT_TYPE,
    include_keys: list[str],
    exclude_keys: list[str] = [],
) -> dict[str, torch.Tensor]:
    target_keys = get_target_keys(
        include_keys,
        exclude_keys,
        list(state_dict.keys()),
    )
    supported_quant_types = ["bnb_nf4", "bnb_fp4", "fp8_e4m3fn"]
    if quant_type not in supported_quant_types:
        raise NotImplementedError("Only bitsandbytes 4bit quantization is supported")

    for key in list(state_dict.keys()):
        if key not in target_keys:
            continue

        if quant_type == "bnb_nf4" or quant_type == "bnb_fp4":
            tensor, state = quantize_4bit(
                state_dict[key].cuda(),
                quant_type=quant_type[len("bnb_") :],  # nf4 or fp4
            )
            state_dict[key] = tensor.cpu()
            for state_key, state_value in state.as_dict(packed=True).items():
                state_dict[f"{key}.{state_key}"] = state_value
        elif quant_type == "fp8_e4m3fn":
            state_dict[key] = state_dict[key].to(torch.float8_e4m3fn)
    return state_dict
