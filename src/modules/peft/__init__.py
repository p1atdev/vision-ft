from pydantic import BaseModel, field_validator

import torch
import torch.nn as nn

from .config import PeftConfigMixin
from .functional import (
    replace_to_peft_layer,
    get_adapter_parameters,
    print_trainable_parameters,
    load_peft_weight,
    while_peft_disabled,
    while_peft_enabled,
)
from .lora import LoRAConfig, LoRALinear, LoRAConv2d
from .loha import LoHaConfig, LoHaLinear

from ...utils.state_dict import RegexMatch


PeftConfigUnion = LoRAConfig | LoHaConfig


class PeftTargetConfig(BaseModel):
    include_keys: list[str | RegexMatch] = []
    exclude_keys: list[str | RegexMatch] = []

    config: PeftConfigUnion

    resume_weight_path: str | None = None
    resume_rename_key_map: dict[str, str] = {}

    @field_validator("include_keys")
    def check_include_keys(cls, v):
        if len(v) == 0:
            raise ValueError("include_keys must not be empty")
        return v

    def replace_to_peft_layer(
        self, model: nn.Module, freeze_base: bool = False
    ) -> None:
        replace_to_peft_layer(
            model,
            self.include_keys,
            self.exclude_keys,
            self.config,
            freeze_base=freeze_base,
        )
