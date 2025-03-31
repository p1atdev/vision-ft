from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file

from .denoiser import Denoiser

from .vae import VAE
from .text_encoder import TextEncoder
from .config import SDXLConfig
# from .scheduler import calculate_time_shift

from .util import convert_to_original_key, convert_from_original_key

from ...modules.quant import replace_by_prequantized_weights
from ...utils import tensor as tensor_utils


class SDXLModel(nn.Module):
    denoiser: Denoiser
    denoiser_class: type[Denoiser] = Denoiser

    def __init__(self, config: SDXLConfig):
        super().__init__()

        self.config = config

        self.denoiser = self.denoiser_class(config.denoiser)
        self.vae = VAE.from_default()
        self.text_encoder = TextEncoder.from_default()

        self.progress_bar = tqdm

    @classmethod
    def from_config(cls, config: SDXLConfig) -> "SDXLModel":
        return cls(config)

    def _from_checkpoint(
        self,
        strict: bool = True,
    ):
        config = self.config
        state_dict = load_file(config.checkpoint_path)
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
        config: SDXLConfig,
    ) -> "SDXLModel":
        with init_empty_weights():
            model = cls.from_config(config)

        model._from_checkpoint()

        return model

    # replace the prefix to match the original checkpoint
    def state_dict(  # type: ignore
        self,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = super().state_dict(
            destination=destination,  # type: ignore
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state_dict = {
            tensor_utils.remove_orig_mod_prefix(key): value
            for key, value in state_dict.items()
        }

        state_dict = {
            convert_to_original_key(key): value for key, value in state_dict.items()
        }

        return state_dict

    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent_channels = self.denoiser.config.in_channels

        if latents is None:
            shape = (
                batch_size,
                latent_channels,
                int(height) // self.vae.compression_ratio,
                int(width) // self.vae.compression_ratio,
            )
            latents = tensor_utils.incremental_seed_randn(
                shape,
                seed=seed,
                dtype=dtype,
                device=device,
            )
        else:
            latents = latents.to(dtype=dtype, device=device)

        return latents

    @torch.no_grad()
    def encode_image(
        self,
        image: Image.Image | list[Image.Image] | torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            image_list = image if isinstance(image, list) else [image]

            image_tensor = tensor_utils.images_to_tensor(
                image_list, self.vae.dtype, self.vae.device
            )
        else:
            image_tensor = image
        encode_output = self.vae.encode(image_tensor.to(self.vae.dtype))
        latents = encode_output[0].sample() * self.vae.scaling_factor

        return latents

    @torch.no_grad()
    def decode_image(
        self,
        latents: torch.Tensor,
    ) -> list[Image.Image]:
        image = self.vae.decode(
            latents / self.vae.scaling_factor,  # type: ignore
            return_dict=False,
        )[0]
        image = tensor_utils.tensor_to_images(image)

        return image
