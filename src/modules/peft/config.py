from typing import Literal
from pydantic import BaseModel, field_validator
from pathlib import Path

from ...utils.state_dict import RegexMatch

PEFT_TYPE = Literal["lora", "none"]


class PeftConfigMixin(BaseModel):
    type: PEFT_TYPE

    dtype: str = "bfloat16"

    include_keys: list[str | RegexMatch] = []
    exclude_keys: list[str | RegexMatch] = []

    resume_weight_path: Path | None = None
    resume_rename_key_map: dict[str, str] = {}

    @field_validator("include_keys")
    def check_include_keys(cls, v):
        if len(v) == 0:
            raise ValueError("include_keys must not be empty")
        return v
