from typing import Literal
from pydantic import BaseModel


PEFT_TYPE = Literal["lora", "none"]


class PeftConfigMixin(BaseModel):
    peft_type: PEFT_TYPE

    include_keys: list[str] = []
    exclude_keys: list[str] = []


class LoRAConfig(PeftConfigMixin):
    peft_type: Literal["lora"] = "lora"
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    use_bias: bool = False
