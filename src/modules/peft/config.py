from typing import Literal
from pydantic import BaseModel


PEFT_TYPE = Literal["lora", "none"]


class PeftConfigMixin(BaseModel):
    type: PEFT_TYPE

    dtype: str = "bfloat16"

    include_keys: list[str] = []
    exclude_keys: list[str] = []
