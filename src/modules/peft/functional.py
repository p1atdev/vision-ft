from typing import Callable
from contextlib import contextmanager
import warnings

import torch
from torch import nn

from .config import PeftConfigMixin, PEFT_TYPE
from .lora import LoRALinear, LoRAConfig
from ...utils.tensor import is_target_key, remove_orig_mod_prefix
from ...utils.dtype import str_to_dtype
from ..quant.functional import collect_children_keys


def _get_peft_linear(
    module: nn.Linear,
    config: PeftConfigMixin,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.type == "none":
        raise ValueError("peft type 'none' is not parameter efficient training")

    if config.type == "lora":
        config = LoRAConfig.model_validate(config.model_dump())
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
                    adapter_params[remove_orig_mod_prefix(f"{name}.{state_key}")] = (
                        state_value
                    )

    return adapter_params


def _load_peft_weight(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    prefix: str = "",
):
    for name, layer in model.named_children():
        full_name = f"{prefix}{name}"

        if isinstance(layer, nn.Linear):
            adapter_state_dict = {
                param_name: state_dict.get(f"{full_name}.{param_name}")
                for param_name in LoRALinear.adapter_weight_names
            }
            if all([value is not None for value in adapter_state_dict.values()]):
                lora_layer = LoRALinear.from_weights(
                    adapter_state_dict,  # type: ignore
                    layer,
                )
                setattr(model, name, lora_layer)

        else:
            _load_peft_weight(
                layer,
                state_dict,
                f"{full_name}.",
            )


def load_peft_weight(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    _load_peft_weight(model, state_dict)


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


def human_readable_param(
    param_size: int,
) -> str:
    # kilo, million, billion, trillion
    units = [
        ("T", 10**12),
        ("B", 10**9),
        ("M", 10**6),
        ("K", 10**3),
    ]
    for unit, value in units:
        if param_size >= value:
            return f"{param_size / value:.2f}{unit}"

    return f"{param_size}"


def print_trainable_parameters(
    model: nn.Module,
    print_fn: Callable = print,
):
    trainable_params = calculate_trainable_parameters(model)
    human_readable_trainable_params = human_readable_param(
        trainable_params["trainable_params"]
    )
    human_readable_all_param = human_readable_param(trainable_params["all_param"])

    print_fn(
        f"Trainable params: {human_readable_trainable_params}, All params: {human_readable_all_param}, Trainable%: {trainable_params['trainable%']:.4f}%"
    )


def set_peft_layer_enabled(model: nn.Module, enabled: bool) -> None:
    for name, module in model.named_modules():
        if hasattr(module, "set_enabled"):
            module.set_enabled(enabled)


@contextmanager
def while_peft_disabled(model: nn.Module):
    """A context manager that temporarily disables all PEFT layers in a model.

    This decorator temporarily disables all Parameter-Efficient Fine-Tuning (PEFT) layers
    in the given model while executing the code within its context, and re-enables them
    afterwards, even if an exception occurs.

    Args:
        model (nn.Module): The PyTorch model containing PEFT layers to be temporarily disabled.

    Yields:
        None: A context manager that can be used in a 'with' statement.

    Example:
        >>> with while_peft_disabled(model):
        ...     # PEFT layers are disabled here
        ...     outputs = model(inputs)
        # PEFT layers are re-enabled here

    Note:
        This is useful when you want to temporarily bypass PEFT modifications
        and use the base model's behavior.
    """
    try:
        set_peft_layer_enabled(model, False)
        yield
    finally:
        set_peft_layer_enabled(model, True)


@contextmanager
def while_peft_enabled(model: nn.Module):
    """
    A context manager that temporarily enables PEFT (Parameter-Efficient Fine-Tuning) layers in a PyTorch model.

    The PEFT layers are enabled upon entering the context and automatically disabled when exiting,
    regardless of whether an exception occurred.

    Args:
        model (nn.Module): The PyTorch model containing PEFT layers to be temporarily enabled.

    Yields:
        None: A context manager that can be used in a 'with' statement.

    Example:
        >>> with while_peft_enabled(model):
        ...     # PEFT layers are enabled here
        ...     output = model(input)
        # PEFT layers are automatically disabled after exiting the context

    Note:
        This ensures PEFT layers are properly disabled even if an exception occurs within the context.
    """
    try:
        set_peft_layer_enabled(model, True)
        yield
    finally:
        set_peft_layer_enabled(model, False)
