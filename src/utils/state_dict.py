import re
from typing import Sequence
from pydantic import BaseModel

import torch


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


def _convert_key_open_clip_to_transformers(key: str) -> str:
    # embeddings
    key = key.replace("positional_embedding", "embeddings.position_embedding.weight", 1)
    key = key.replace("token_embedding", "embeddings.token_embedding", 1)

    # transformer blocks
    key = key.replace("transformer.resblocks", "encoder.layers", 1)
    key = key.replace(".attn.", ".self_attn.", 1)
    key = key.replace(".ln_1.", ".layer_norm1.", 1)
    key = key.replace(".ln_2.", ".layer_norm2.", 1)
    key = key.replace(".mlp.c_fc.", ".mlp.fc1.", 1)
    key = key.replace(".mlp.c_proj.", ".mlp.fc2.", 1)
    # key = key.replace(".out_proj.", ".out_proj.", 1) # as is

    # final
    key = key.replace("ln_final", "final_layer_norm", 1)

    return key


def _convert_key_transformers_to_open_clip(key: str) -> str:
    # embeddings
    key = key.replace("embeddings.position_embedding.weight", "positional_embedding", 1)
    key = key.replace("embeddings.token_embedding", "token_embedding", 1)

    # transformer blocks
    key = key.replace("encoder.layers", "transformer.resblocks", 1)
    key = key.replace(".self_attn.", ".attn.", 1)
    key = key.replace(".layer_norm1.", ".ln_1.", 1)
    key = key.replace(".layer_norm2.", ".ln_2.", 1)
    key = key.replace(".mlp.fc1.", ".mlp.c_fc.", 1)
    key = key.replace(".mlp.fc2.", ".mlp.c_proj.", 1)
    # key = key.replace(".out_proj.", ".out_proj.", 1) # as is

    # final
    key = key.replace("final_layer_norm", "ln_final", 1)

    return key


def _in_proj_weight_to_qkv(
    in_proj_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert in_proj_weight.size(0) % 3 == 0, (
        "in_proj_weight shape must be divisible by 3"
    )
    to_q, to_k, to_v = in_proj_weight.chunk(3, dim=0)
    return (
        to_q,
        to_k,
        to_v,
    )


def _qkv_to_in_proj_weight(
    to_q: torch.Tensor,
    to_k: torch.Tensor,
    to_v: torch.Tensor,
) -> torch.Tensor:
    assert to_q.size(0) == to_k.size(0) == to_v.size(0), (
        "to_q, to_k, to_v must have the same size"
    )
    in_proj_weight = torch.cat([to_q, to_k, to_v], dim=0)
    return in_proj_weight


def _in_proj_bias_to_qkv(
    in_proj_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert in_proj_bias.size(0) % 3 == 0, "in_proj_bias shape must be divisible by 3"
    to_q, to_k, to_v = in_proj_bias.chunk(3, dim=0)
    return (
        to_q,
        to_k,
        to_v,
    )


def _qkv_to_in_proj_bias(
    to_q: torch.Tensor,
    to_k: torch.Tensor,
    to_v: torch.Tensor,
) -> torch.Tensor:
    assert to_q.size(0) == to_k.size(0) == to_v.size(0), (
        "to_q, to_k, to_v must have the same size"
    )
    in_proj_bias = torch.cat([to_q, to_k, to_v], dim=0)
    return in_proj_bias


def convert_open_clip_to_transformers(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        if "logit_scale" in key:
            # skip logit_scale
            continue
        new_key = _convert_key_open_clip_to_transformers(key)
        new_state_dict[new_key] = value

    for key in list(new_state_dict.keys()):
        if re.match(r".*\.in_proj_weight$", key):
            value = new_state_dict[key]
            to_q, to_k, to_v = _in_proj_weight_to_qkv(value)
            new_state_dict[key.replace("in_proj_weight", "q_proj.weight")] = to_q
            new_state_dict[key.replace("in_proj_weight", "k_proj.weight")] = to_k
            new_state_dict[key.replace("in_proj_weight", "v_proj.weight")] = to_v
            del new_state_dict[key]
        elif re.match(r".*\.in_proj_bias$", key):
            value = new_state_dict[key]
            to_q, to_k, to_v = _in_proj_bias_to_qkv(value)
            new_state_dict[key.replace("in_proj_bias", "q_proj.bias")] = to_q
            new_state_dict[key.replace("in_proj_bias", "k_proj.bias")] = to_k
            new_state_dict[key.replace("in_proj_bias", "v_proj.bias")] = to_v
            del new_state_dict[key]

    return new_state_dict


def convert_transformers_to_open_clip(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    new_state_dict = {}

    for key, value in state_dict.items():
        if groups := re.search(r"(.*)\.(q|k|v)_proj\.(weight|bias)$", key):
            base_key = groups.group(1)
            # print("found", key)
            to_q_weight = state_dict[f"{base_key}.q_proj.weight"]
            to_k_weight = state_dict[f"{base_key}.k_proj.weight"]
            to_v_weight = state_dict[f"{base_key}.v_proj.weight"]
            new_state_dict[
                _convert_key_transformers_to_open_clip(f"{base_key}.in_proj_weight")
            ] = _qkv_to_in_proj_weight(to_q_weight, to_k_weight, to_v_weight)

            to_q_bias = state_dict[f"{base_key}.q_proj.bias"]
            to_k_bias = state_dict[f"{base_key}.k_proj.bias"]
            to_v_bias = state_dict[f"{base_key}.v_proj.bias"]
            new_state_dict[
                _convert_key_transformers_to_open_clip(f"{base_key}.in_proj_bias")
            ] = _qkv_to_in_proj_bias(to_q_bias, to_k_bias, to_v_bias)
        else:
            new_key = _convert_key_transformers_to_open_clip(key)
            new_state_dict[new_key] = value

    return new_state_dict
