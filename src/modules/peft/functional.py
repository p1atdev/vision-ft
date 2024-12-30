import torch
from torch import nn

from .config import PeftConfigMixin, PEFT_TYPE, LoRAConfig
from .lora import LoRALinear
from ...utils.tensor import is_target_key


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
    dtype: torch.dtype | None = None,
) -> None:
    for name, layer in model.named_children():
        full_name = f"{prefix}{name}"

        if isinstance(layer, nn.Linear):
            if not is_target_key(full_name, config.include_keys, config.exclude_keys):
                continue

            # replace with peft module
            peft_module = _get_peft_linear(layer, config, dtype=dtype)
            setattr(model, name, peft_module)
        else:
            _replace_to_peft_linear(
                layer,
                config,
                f"{full_name}.",
                dtype=dtype,
            )


def replace_to_peft_linear(
    model: nn.Module,
    config: PeftConfigMixin,
    dtype: torch.dtype | None = None,
) -> None:
    _replace_to_peft_linear(
        model,
        config,
        dtype=dtype,
    )
