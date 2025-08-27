import re
from typing import Literal

# root

# Text encoder


def text_encoder_convert_from_original_key(key: str) -> str:
    if not key.startswith("model."):
        return f"model.{key}"

    return key


def text_encoder_convert_to_original_key(key: str) -> str:
    if key.startswith("model."):
        return key[6:]  # Remove "model." prefix
    return key


# Denoiser
def denoiser_convert_from_original_key(key: str) -> str:
    if key.startswith("model."):
        return key[6:]  # Remove "model." prefix

    return key


def denoiser_convert_to_original_key(key: str) -> str:
    if not key.startswith("model."):
        return f"model.{key}"

    return key


# vae
def vae_convert_from_original_key(key: str) -> str:
    return key


def vae_convert_to_original_key(key: str) -> str:
    return key


# auto
def convert_from_original_key(
    key: str, module: Literal["text_encoder", "denoiser", "vae"]
) -> str:
    if module == "text_encoder":
        return text_encoder_convert_from_original_key(key)
    elif module == "denoiser":
        return denoiser_convert_from_original_key(key)
    elif module == "vae":
        return vae_convert_from_original_key(key)
    else:
        raise ValueError(f"Unknown module: {module}")


def convert_to_original_key(
    key: str, module: Literal["text_encoder", "denoiser", "vae"]
) -> str:
    if module == "text_encoder":
        return text_encoder_convert_to_original_key(key)
    elif module == "denoiser":
        return denoiser_convert_to_original_key(key)
    elif module == "vae":
        return vae_convert_to_original_key(key)
    else:
        raise ValueError(f"Unknown module: {module}")
