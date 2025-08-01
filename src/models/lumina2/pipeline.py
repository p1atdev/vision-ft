from tqdm import tqdm
from PIL import Image
import warnings

import torch
from torch._tensor import Tensor
import torch.nn as nn

from safetensors.torch import load_file
from accelerate import init_empty_weights

from .config import Lumina2Config
from .denoiser import (
    Denoiser,
)
from .text_encoder import (
    TextEncoder,
)
from .util import convert_from_original_key
from .vae import VAE
from ...utils import tensor as tensor_utils
from ...modules.quant import replace_by_prequantized_weights


class Lumina2(nn.Module):
    denoiser: Denoiser
    denoiser_class: type[Denoiser] = Denoiser

    def __init__(self, config: Lumina2Config):
        super().__init__()

        self.config = config

        self._setup_models(config)

    def _setup_models(self, config):
        self.denoiser = self.denoiser_class(config.denoiser)
        self.vae = VAE.from_default()
        self.text_encoder = TextEncoder.from_default()

        # TODO
        # self.scheduler = Scheduler()  # euler

        self.progress_bar = tqdm

    @classmethod
    def from_config(cls, config: Lumina2Config) -> "Lumina2":
        return cls(config)

    def _from_checkpoint(
        self,
        strict: bool = True,
    ):
        config = self.config
        state_dict = load_file(config.checkpoint_path)
        # TODO
        state_dict = {
            convert_from_original_key(key): value for key, value in state_dict.items()
        }

        # prepare for prequantized weights
        replace_by_prequantized_weights(self, state_dict)

        self.denoiser.load_state_dict(
            {
                key[len("denoiser.") :]: value
                for key, value in state_dict.items()
                if key.startswith("denoiser.")
            },
            strict=strict,
            assign=True,
        )
        self.vae.load_state_dict(
            {
                key[len("vae.") :]: value
                for key, value in state_dict.items()
                if key.startswith("vae.")
            },
            strict=strict,
            assign=True,
        )
        self.text_encoder.load_state_dict(
            {
                key[len("text_encoder.") :]: value
                for key, value in state_dict.items()
                if key.startswith("text_encoder.")
            },
            strict=strict,
            assign=True,
        )

    @classmethod
    def from_checkpoint(
        cls,
        config: Lumina2Config,
    ) -> "Lumina2":
        with init_empty_weights():
            model = cls.from_config(config)

        model._from_checkpoint()

        return model
