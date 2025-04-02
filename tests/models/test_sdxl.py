import os
import tempfile
from huggingface_hub import hf_hub_download

import numpy as np
import torch
from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteScheduler

from src.models.sdxl.util import (
    unet_block_convert_from_original_key,
    unet_block_convert_to_original_key,
    vae_convert_from_original_key,
    vae_convert_to_original_key,
    convert_from_original_key,
    convert_to_original_key,
)
from src.models.sdxl import SDXLConfig, SDXLModel, DenoiserConfig
from src.models.sdxl.scheduler import Scheduler


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
            encoder_hidden_state = torch.randn(1, 77, 2048, device="cuda")
            encoder_pooler_output = torch.randn(1, 1280, device="cuda")
            original_size = torch.tensor([128, 128], device="cuda").unsqueeze(0)
            target_size = torch.tensor([128, 128], device="cuda").unsqueeze(0)
            crop_coords_top_left = torch.tensor([0, 0], device="cuda").unsqueeze(0)

            output = model.denoiser(
                latents=latent,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_state,
                encoder_pooler_output=encoder_pooler_output,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

    assert output is not None, "Output is None"
    assert output.shape == latent.shape, "Output shape does not match input shape"


def test_euler_scheduler():
    reference = EulerDiscreteScheduler.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        subfolder="scheduler",
    )
    scheduler = Scheduler()

    test_cases = [
        #  num_inference_steps
        17,
        20,
        28,
        50,
    ]

    for num_inference_steps in test_cases:
        timesteps = scheduler.get_timesteps(num_inference_steps)
        sigmas = scheduler.get_sigmas(timesteps)

        reference.set_timesteps(num_inference_steps)

        assert timesteps.shape == reference.timesteps.shape, (
            f"Timesteps shape mismatch for {num_inference_steps}",
            timesteps.shape,
            reference.timesteps.shape,
        )
        assert sigmas.shape == reference.sigmas.shape, (
            f"Sigmas shape mismatch for {num_inference_steps}",
            sigmas.shape,
            reference.sigmas.shape,
        )


def test_generate_illustrious_xl():
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
            images = model.generate(
                prompt="1girl, solo, masterpiece, best quality",
                negative_prompt="worst quality, low quality",
                cfg_scale=5.0,
                execution_dtype=torch.bfloat16,
                device="cuda:0",
                do_offloading=False,
            )

    with tempfile.TemporaryDirectory(delete=False) as temp_file:
        temp_file = os.path.join(temp_file, "test.webp")
        images[0].save(temp_file)
        print(f"Image saved to {temp_file}")
