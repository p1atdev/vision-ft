from typing import Callable

import torch
from torch import nn

from .config import PeftConfigMixin, PEFT_TYPE
from .lora import LoRALinear, LoRAConfig
from ...utils.tensor import is_target_key
from ...utils.dtype import str_to_dtype


def _get_peft_linear(
    module: nn.Linear,
    config: PeftConfigMixin,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.type == "none":
        raise ValueError("peft type 'none' is not parameter efficient training")

    if isinstance(config, LoRAConfig):
        return LoRALinear(
            config=config,
            original_linear=module,
            in_features=module.in_features,
            out_features=module.out_features,
            dropout=config.dropout,
            dtype=dtype,
        )

    else:
        raise ValueError(f"Unknown peft type: {config.type}")


def _replace_to_peft_linear(
    model: nn.Module,
    config: PeftConfigMixin,
    prefix: str = "",
) -> None:
    for name, layer in model.named_children():
        full_name = f"{prefix}{name}"

        # disable grad for all layers
        layer.requires_grad_(False)

        if isinstance(layer, nn.Linear):
            if not is_target_key(full_name, config.include_keys, config.exclude_keys):
                continue

            # replace with peft module
            peft_module = _get_peft_linear(
                layer, config, dtype=str_to_dtype(config.dtype)
            )
            setattr(model, name, peft_module)
        else:
            _replace_to_peft_linear(
                layer,
                config,
                f"{full_name}.",
            )


def replace_to_peft_linear(
    model: nn.Module,
    config: PeftConfigMixin,
) -> None:
    _replace_to_peft_linear(
        model,
        config,
    )


def get_adapter_parameters(model: nn.Module) -> dict[str, torch.Tensor]:
    adapter_params = {}

    for name, module in model.named_modules():
        if (param_names := getattr(module, "adapter_param_names", None)) is not None:
            for state_key, state_value in module.state_dict().items():
                if any(state_key.startswith(name) for name in param_names):
                    adapter_params[f"{name}.{state_key}"] = state_value

    return adapter_params


# ref: https://github.com/huggingface/peft/blob/6d458b300fc2ed82e19f796b53af4c97d03ea604/examples/fp4_finetuning/finetune_fp4_opt_bnb_peft.py#L92-L104
def calculate_trainable_parameters(
    model: nn.Module,
):
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return {
        "trainable_params": trainable_params,
        "all_param": all_param,
        "trainable%": 100 * trainable_params / all_param,
    }


def print_trainable_parameters(
    model: nn.Module,
    print_fn: Callable = print,
):
    trainable_params = calculate_trainable_parameters(model)
    print_fn(
        f"Trainable params: {trainable_params['trainable_params']}, All params: {trainable_params['all_param']}, Trainable%: {trainable_params['trainable%']:.2f}%"
    )
