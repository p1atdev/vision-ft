import re
from typing import Sequence
from pydantic import BaseModel


class RegexMatch(BaseModel):
    regex: str

    def __call__(self, value: str) -> bool:
        return bool(re.match(self.regex, value))


def get_target_keys(
    include: Sequence[str | RegexMatch],
    exclude: Sequence[str | RegexMatch],
    keys: list[str],
) -> list[str]:
    matched_keys = set()

    # include keys
    for pattern in include:
        if isinstance(pattern, str):
            # is pattern in the key?
            matched_keys.update([key for key in keys if pattern in key])
        elif isinstance(pattern, RegexMatch):
            _pattern = re.compile(pattern.regex)
            # is pattern matched in the key?
            matched_keys.update([key for key in keys if _pattern.match(key)])

    # remove exclude keys
    for pattern in exclude:
        if isinstance(pattern, str):
            # is pattern in the key?
            matched_keys.difference_update([key for key in keys if pattern in key])
        elif isinstance(pattern, RegexMatch):
            _pattern = re.compile(pattern.regex)
            # is pattern matched in the key?
            matched_keys.difference_update([key for key in keys if _pattern.match(key)])

    return list(matched_keys)
