import os
from huggingface_hub import hf_hub_download

import torch

from src.models.sdxl.util import (
    unet_block_convert_from_original_key,
    unet_block_convert_to_original_key,
    vae_convert_from_original_key,
    vae_convert_to_original_key,
    convert_from_original_key,
    convert_to_original_key,
)
from src.models.sdxl import SDXLConfig, SDXLModel, DenoiserConfig


def test_unet_block_key():
    test_cases = [
        # original, custom
        (
            "input_blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight",
            "input_blocks.blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight",
        ),
        (
            "input_blocks.3.0.op.bias",
            "input_blocks.blocks.3.0.op.bias",
        ),
        (
            "output_blocks.7.0.in_layers.0.weight",
            "output_blocks.blocks.7.0.in_layers.0.weight",
        ),
        (
            "output_blocks.8.0.skip_connection.weight",
            "output_blocks.blocks.8.0.skip_connection.weight",
        ),
        (
            "middle_block.1.transformer_blocks.0.attn1.to_k.weight",
            "middle_block.blocks.1.transformer_blocks.0.attn1.to_k.weight",
        ),
    ]

    for input, expected in test_cases:
        assert unet_block_convert_from_original_key(input) == expected, (
            input,
            unet_block_convert_from_original_key(input),
            expected,
        )

    for expected, input in test_cases:
        assert unet_block_convert_to_original_key(input) == expected


def test_vae_block_key():
    test_cases = [
        # original, custom
        (
            "encoder.down.1.block.0.conv1.weight",
            "encoder.down_blocks.1.resnets.0.conv1.weight",
        ),
        (
            "encoder.down.2.block.0.nin_shortcut.weight",
            "encoder.down_blocks.2.resnets.0.conv_shortcut.weight",
        ),
        ("decoder.mid.attn_1.q.weight", "decoder.mid_block.attentions.0.to_q.weight"),
        (
            "decoder.up.0.block.0.nin_shortcut.weight",
            "decoder.up_blocks.3.resnets.0.conv_shortcut.weight",
        ),
        ("post_quant_conv", "post_quant_conv"),
        ("encoder.norm_out.weight", "encoder.conv_norm_out.weight"),
        ("decoder.conv_in.weight", "decoder.conv_in.weight"),
    ]

    for input, expected in test_cases:
        assert vae_convert_from_original_key(input) == expected

    for expected, input in test_cases:
        assert vae_convert_to_original_key(input) == expected


def test_convert_key():
    test_cases = [
        # original, custom
        # unet
        (
            "model.diffusion_model.input_blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight",
            "denoiser.input_blocks.blocks.5.1.transformer_blocks.1.attn1.to_out.0.weight",  # no change
        ),
        (
            "model.diffusion_model.input_blocks.3.0.op.bias",
            "denoiser.input_blocks.blocks.3.0.op.bias",
        ),
        (
            "model.diffusion_model.output_blocks.7.0.in_layers.0.weight",
            "denoiser.output_blocks.blocks.7.0.in_layers.0.weight",
        ),
        (
            "model.diffusion_model.output_blocks.8.0.skip_connection.weight",
            "denoiser.output_blocks.blocks.8.0.skip_connection.weight",
        ),
        (
            "model.diffusion_model.middle_block.1.transformer_blocks.0.attn1.to_k.weight",
            "denoiser.middle_block.blocks.1.transformer_blocks.0.attn1.to_k.weight",
        ),
        # text encoder
        (
            "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight",
            "text_encoder.text_encoder_1.text_model.embeddings.position_embedding.weight",
        ),
        (  # this is not perfect transformation at this time, and will be renamed at convert_open_clip_to_transformers
            "conditioner.embedders.1.model.transformer.resblocks.0.attn.in_proj_bias",
            "text_encoder.text_encoder_2.text_model.transformer.resblocks.0.attn.in_proj_bias",
        ),
        (
            "conditioner.embedders.1.model.text_projection",
            "text_encoder.text_encoder_2.text_projection.weight",
        ),
        # vae
        (
            "first_stage_model.encoder.down.1.block.0.conv1.weight",
            "vae.encoder.down_blocks.1.resnets.0.conv1.weight",
        ),
        (
            "first_stage_model.encoder.down.2.block.0.nin_shortcut.weight",
            "vae.encoder.down_blocks.2.resnets.0.conv_shortcut.weight",
        ),
        (
            "first_stage_model.decoder.mid.attn_1.q.weight",
            "vae.decoder.mid_block.attentions.0.to_q.weight",
        ),
        (
            "first_stage_model.decoder.up.0.block.0.nin_shortcut.weight",
            "vae.decoder.up_blocks.3.resnets.0.conv_shortcut.weight",
        ),
        ("first_stage_model.post_quant_conv", "vae.post_quant_conv"),
        (
            "first_stage_model.encoder.norm_out.weight",
            "vae.encoder.conv_norm_out.weight",
        ),
        ("first_stage_model.decoder.conv_in.weight", "vae.decoder.conv_in.weight"),
    ]

    for input, expected in test_cases:
        assert convert_from_original_key(input) == expected

    for expected, input in test_cases:
        assert convert_to_original_key(input) == expected


def test_load_illustrious_xl():
    repo_name = "OnomaAIResearch/Illustrious-XL-v1.1"
    path = hf_hub_download(
        repo_name,
        filename="Illustrious-XL-v1.1.safetensors",
    )
    assert os.path.exists(path), f"File {path} does not exist"

    config = SDXLConfig(
        checkpoint_path=path,
        pretrained_model_name_or_path=repo_name,
        denoiser=DenoiserConfig(attention_backend="xformers"),
    )

    model = SDXLModel.from_checkpoint(config)
    model.to(device="cuda")

    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent = torch.randn(1, 4, 128, 128, device="cuda")
            timesteps = torch.randint(0, 1000, (1,), device="cuda")
            context = torch.randn(1, 77, 2048, device="cuda")
            output = model.denoiser(latent, timesteps, context)

    assert output is not None, "Output is None"
    assert output.shape == latent.shape, "Output shape does not match input shape"
