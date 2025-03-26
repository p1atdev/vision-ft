from src.models.sdxl.util import (
    unet_block_convert_from_original_key,
    unet_block_convert_to_original_key,
    convert_from_original_key,
    convert_to_original_key,
)


def test_unet_block_key():
    test_cases = [
        # original, custom
        ("middle_block.0.emb_layers.1.bias", "middle_block.blocks.0.emb_layers.1.bias"),
        ("middle_block.1.proj_in.weight", "middle_block.blocks.1.proj_in.weight"),
        ("input_blocks.0.0.bias", "input_blocks.in_conv.bias"),
        (
            "input_blocks.1.0.emb_layers.1.weight",
            "input_blocks.blocks.0.emb_layers.1.weight",
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


def test_convert_key():
    test_cases = [
        # original, custom
        (
            "diffusion_model.middle_block.0.emb_layers.1.bias",
            "denoiser.middle_block.blocks.0.emb_layers.1.bias",
        ),
        (
            "diffusion_model.middle_block.1.proj_in.weight",
            "denoiser.middle_block.blocks.1.proj_in.weight",
        ),
    ]

    for input, expected in test_cases:
        assert convert_from_original_key(input) == expected

    for expected, input in test_cases:
        assert convert_to_original_key(input) == expected
