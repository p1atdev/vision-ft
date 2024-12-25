from pathlib import Path


import torch.nn as nn

from transformers import AutoTokenizer
from safetensors.torch import load_file
from accelerate import init_empty_weights

from .config import AuraFlowConig
from .denoiser import (
    Denoiser,
    DENOISER_TENSOR_PREFIX,
)
from .text_encoder import (
    TextEncoder,
    DEFAULT_TEXT_ENCODER_CONFIG,
    DEFAULT_TEXT_ENCODER_CLASS,
    DEFAULT_TEXT_ENCODER_CONFIG_CLASS,
    DEFAULT_TOKENIZER_REPO,
    DEFAULT_TOKENIZER_FOLDER,
    TEXT_ENCODER_TENSOR_PREFIX,
)
from .vae import VAE, DEFAULT_VAE_CONFIG


class AuraFlowForTraining(nn.Module):
    def __init__(self, config: AuraFlowConig):
        super().__init__()

        self.denoiser = Denoiser.from_config(config.denoiser_config)
        vae = VAE.from_config(DEFAULT_VAE_CONFIG)
        assert isinstance(vae, VAE)
        self.vae = vae
        _text_encoder = DEFAULT_TEXT_ENCODER_CLASS._from_config(
            DEFAULT_TEXT_ENCODER_CONFIG_CLASS(**DEFAULT_TEXT_ENCODER_CONFIG),
        )
        _tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER_REPO, subfolder=DEFAULT_TOKENIZER_FOLDER
        )
        self.text_encoder = TextEncoder(model=_text_encoder, tokenizer=_tokenizer)

    def forward(self, x):
        return self.denoiser(x)

    @classmethod
    def from_config(cls, config: AuraFlowConig) -> "AuraFlowForTraining":
        return cls(config)

    @classmethod
    def from_pretrained(cls, config: AuraFlowConig) -> "AuraFlowForTraining":
        with init_empty_weights():
            model = cls.from_config(config)

        state_dict = load_file(config.checkpoint_path)
        model.denoiser.load_state_dict(
            {
                key[len(DENOISER_TENSOR_PREFIX) :]: value
                for key, value in state_dict.items()
                if key.startswith(DENOISER_TENSOR_PREFIX)
            },
            assign=True,
        )
        model.vae = VAE.from_pretrained(
            config.pretrained_model_name_or_path,
            subfolder=config.vae_folder,
        )
        model.text_encoder.model.load_state_dict(
            {
                key[len(TEXT_ENCODER_TENSOR_PREFIX) :]: value
                for key, value in state_dict.items()
                if key.startswith(TEXT_ENCODER_TENSOR_PREFIX)
            },
            assign=True,
        )

        return model


def load_models(
    config: AuraFlowConig,
) -> AuraFlowForTraining:
    # model_name_or_path = config.pretrained_model_name_or_path

    # if (path := Path(model_name_or_path)).exists():
    #     if path.is_file():
    return AuraFlowForTraining.from_pretrained(config)

    # raise NotImplementedError("Only from single file is supported for now.")

    # return load_from_folder(config)
