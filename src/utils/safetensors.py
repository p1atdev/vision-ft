import torch
from pathlib import Path

from safetensors.torch import load_file


def load_file_with_rename_key_map(
    file_path: str | Path, rename_key_map: dict[str, str]
):
    state_dict = load_file(file_path)

    def replace(key: str):
        for prefix, to in rename_key_map.items():
            key = key.replace(prefix, to, 1)
        return key

    state_dict = {replace(k): v for k, v in state_dict.items()}
    return state_dict
