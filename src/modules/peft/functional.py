import torch
from torch import nn

from .config import PeftConfigMixin, PEFT_TYPE, LoRAConfig
from .lora import LoRALinear
from ...utils.tensor import is_target_key
from ...utils.dtype import str_to_dtype


def _get_peft_linear(
    module: nn.Linear,
    config: PeftConfigMixin,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    if config.peft_type == "none":
        raise ValueError("peft_type 'none' is not parameter efficient training")

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
        raise ValueError(f"Unknown peft_type: {config.peft_type}")


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
