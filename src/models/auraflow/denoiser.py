import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.transformers.auraflow_transformer_2d import (
    AuraFlowTransformer2DModel,
)

DEFAULT_DENOISER_CONFIG = {
    "attention_head_dim": 256,
    "caption_projection_dim": 3072,
    "in_channels": 4,
    "joint_attention_dim": 2048,
    "num_attention_heads": 12,
    "num_mmdit_layers": 4,
    "num_single_dit_layers": 32,
    "out_channels": 4,
    "patch_size": 2,
    "pos_embed_max_size": 9216,
    "sample_size": 64,
}
DENOISER_TENSOR_PREFIX = "model."


class Denoiser(
    AuraFlowTransformer2DModel,
    nn.Module,
):
    pass
