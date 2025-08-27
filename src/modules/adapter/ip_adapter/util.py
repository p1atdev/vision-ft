from typing import Literal

import torch.nn as nn


NORMALIZATION_TYPES = Literal["layernorm", "layer", "rmsnorm", "rms"]


def get_norm_layer(
    normalization: NORMALIZATION_TYPES,
    **kwargs,
) -> nn.Module:
    """
    Get the normalization layer based on the normalization type.
    """
    if normalization.lower() in ["layernorm", "layer"]:
        return nn.LayerNorm(**kwargs)
    elif normalization.lower() in ["rmsnorm", "rms"]:
        return nn.RMSNorm(**kwargs)
    else:
        raise ValueError(f"Unsupported normalization type: {normalization}")
