from typing import Literal
from pydantic import BaseModel

PEFT_TYPE = Literal["lora", "loha", "none"]


class PeftConfigMixin(BaseModel):
    type: PEFT_TYPE

    dtype: str = "bfloat16"
