from .config import PeftConfigMixin
from .lora import LoRALinear, LoRAConfig
from .functional import (
    replace_to_peft_layer,
    get_adapter_parameters,
    print_trainable_parameters,
    load_peft_weight,
    while_peft_disabled,
    while_peft_enabled,
)

PeftConfigUnion = LoRAConfig
