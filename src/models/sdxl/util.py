import re

# MARK: Key name renaming


def unet_block_convert_from_original_key(key: str) -> str:
    # (input|output)_blocks. -> (input|output)_blocks.blocks.
    key = re.sub(
        r"(input|output)_blocks\.",
        r"\1_blocks.blocks.",
        key,
    )
    key = key.replace("middle_block.", "middle_block.blocks.", 1)

    return key


def unet_block_convert_to_original_key(key: str) -> str:
    # (input|output)_blocks.blocks. -> (input|output)_blocks.
    key = re.sub(
        r"(input|output)_blocks\.blocks\.",
        r"\1_blocks.",
        key,
    )
    key = key.replace("middle_block.blocks.", "middle_block.", 1)

    return key


def denoiser_convert_from_original_key(key: str) -> str:
    key = unet_block_convert_from_original_key(key)

    return key


def denoiser_convert_to_original_key(key: str) -> str:
    key = unet_block_convert_to_original_key(key)

    return key


def root_convert_from_original_key(key: str) -> str:
    # denoiser
    key = key.replace("model.diffusion_model.", "diffusion_model.", 1)
    key = key.replace("diffusion_model.", "denoiser.", 1)

    # text_encoder
    key = key.replace(
        "conditioner.embedders.0.transformer.", "text_encoder.text_encoder_1.", 1
    )
    key = key.replace(
        "conditioner.embedders.1.model.text_projection",
        "text_encoder.text_encoder_2.text_projection.weight",
        1,
    )  # WithProjection
    key = key.replace(
        "conditioner.embedders.1.model.",
        "text_encoder.text_encoder_2.text_model.",
        1,
    )  # CLIPTextModel

    # vae
    # TODO

    return key


def root_convert_to_original_key(key: str) -> str:
    # denoiser
    key = key.replace("denoiser.", "model.diffusion_model.", 1)

    # text_encoder
    key = key.replace(
        "text_encoder.text_encoder_1.", "conditioner.embedders.0.transformer.", 1
    )
    key = key.replace(
        "text_encoder.text_encoder_2.text_projection.weight",
        "conditioner.embedders.1.model.text_projection",
        1,
    )  # WithProjection
    key = key.replace(
        "text_encoder.text_encoder_2.text_model.",
        "conditioner.embedders.1.model.",
        1,
    )  # CLIPTextModel

    # vae
    # TODO

    return key


def convert_from_original_key(key: str) -> str:
    key = root_convert_from_original_key(key)

    key = denoiser_convert_from_original_key(key)

    return key


def convert_to_original_key(key: str) -> str:
    key = root_convert_to_original_key(key)

    key = denoiser_convert_to_original_key(key)

    return key
