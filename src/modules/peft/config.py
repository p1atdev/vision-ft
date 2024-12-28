from typing import Literal
from pydantic import BaseModel

from ...utils.quantize import QUANT_TYPE

PEFT_TYPE = Literal["lora", "qlora", "none"]


class PeftConfigMixin(BaseModel):
    peft_type: PEFT_TYPE


class LoRAConfig(PeftConfigMixin):
    peft_type: Literal["lora"] = "lora"
    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    use_bisa: bool = False

    include_keys: list[str] = []
    exclude_keys: list[str] = []


class QLoRAConfig(LoRAConfig):
    peft_type: Literal["qlora"] = "qlora"
    quant_type: QUANT_TYPE = "nf4"

    quant_args: dict = {}
