from pathlib import Path


from transformers import AutoTokenizer, AutoModelForTextEncoding
from safetensors.torch import load_file

from .config import AuraFlowConig
from .denoiser import Denoiser, DEFAULT_DENOISER_CONFIG, DENOISER_TENSOR_PREFIX
from .text_encoder import (
    TextEncoder,
    DEFAULT_TEXT_ENCODER_CONFIG,
    DEFAULT_TEXT_ENCODER_CLASS,
    DEFAULT_TOKENIZER_REPO,
    DEFAULT_TOKENIZER_FOLDER,
    TEXT_ENCODER_TENSOR_PREFIX,
)
from .vae import VAE, DEFAULT_VAE_CONFIG, VAE_TENSOR_PREFIX


def load_from_single_file(
    config: AuraFlowConig,
):
    denoiser = Denoiser.from_config(DEFAULT_DENOISER_CONFIG, device_map="auto")
    assert isinstance(denoiser, Denoiser)
    vae = VAE.from_config(DEFAULT_VAE_CONFIG, device_map="auto")
    assert isinstance(vae, VAE)

    _text_encoder = DEFAULT_TEXT_ENCODER_CLASS._from_config(
        DEFAULT_TEXT_ENCODER_CONFIG, device_map="auto"
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        DEFAULT_TOKENIZER_REPO, subfolder=DEFAULT_TOKENIZER_FOLDER
    )

    state_dict = load_file(config.pretrained_model_name_or_path)
    denoiser.load_state_dict(
        {
            key: value
            for key, value in state_dict.items()
            if key.startswith(DENOISER_TENSOR_PREFIX)
        }
    )
    vae.load_state_dict(
        {
            key: value
            for key, value in state_dict.items()
            if key.startswith(VAE_TENSOR_PREFIX)
        }
    )
    _text_encoder.load_state_dict(
        {
            key: value
            for key, value in state_dict.items()
            if key.startswith(TEXT_ENCODER_TENSOR_PREFIX)
        }
    )

    text_encoder = TextEncoder(model=_text_encoder, tokenizer=_tokenizer)

    return denoiser, vae, text_encoder


def load_from_folder(config: AuraFlowConig) -> tuple[Denoiser, VAE, TextEncoder]:
    folder = config.pretrained_model_name_or_path
    denoiser = Denoiser.from_pretrained(
        folder,
        subfolder=config.denoiser_folder,
        torch_dtype=config.get_dtype(),
        variant=config.variant,
    )
    assert isinstance(denoiser, Denoiser)
    vae = VAE.from_pretrained(
        folder,
        subfolder=config.vae_folder,
        torch_dtype=config.get_dtype(),
        variant=config.variant,
    )
    assert isinstance(vae, VAE)
    text_encoder = TextEncoder(
        model=AutoModelForTextEncoding.from_pretrained(
            folder,
            subfolder=config.text_encoder_folder,
            torch_dtype=config.get_dtype(),
            variant=config.variant,
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            folder,
            subfolder=config.tokenizer_folder,
        ),
    )
    assert isinstance(text_encoder, TextEncoder)

    return denoiser, vae, text_encoder


def load_models(
    config: AuraFlowConig,
) -> tuple[Denoiser, VAE, TextEncoder]:
    model_name_or_path = config.pretrained_model_name_or_path

    if (path := Path(model_name_or_path)).exists():
        if path.is_file():
            return load_from_single_file(config)

    return load_from_folder(config)
