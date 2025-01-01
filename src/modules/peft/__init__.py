from .config import PeftConfigMixin
from .lora import LoRALinear, LoRAConfig
from .functional import (
    replace_to_peft_linear,
    get_adapter_parameters,
    print_trainable_parameters,
)

PeftConfigUnion = LoRAConfig
