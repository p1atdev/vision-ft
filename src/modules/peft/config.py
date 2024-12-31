from typing import Literal
from pydantic import BaseModel, field_validator


PEFT_TYPE = Literal["lora", "none"]


class PeftConfigMixin(BaseModel):
    type: PEFT_TYPE

    dtype: str = "bfloat16"

    include_keys: list[str] = []
    exclude_keys: list[str] = []

    @field_validator("include_keys")
    def check_include_keys(cls, v):
        if len(v) == 0:
            raise ValueError("include_keys must not be empty")
        return v
