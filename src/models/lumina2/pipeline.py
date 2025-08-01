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
from .text_encoder import TextEncoder, TextEncodingOutput
from .vae import VAE
from .scheduler import Scheduler
from .util import convert_from_original_key
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

        self.scheduler = Scheduler()  # euler

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

    # MARK: prepare_latents
    def _prepare_latents(
        self,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None = None,
    ) -> torch.Tensor:
        latent_channels = self.denoiser.config.in_channels

        shape = (
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

        return latents

    def prepare_nested_latents(
        self,
        heights: list[int],
        widths: list[int],
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None = None,
    ):
        latents = torch.nested.as_nested_tensor(
            [
                self._prepare_latents(
                    height=height,
                    width=width,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                )
                for height, width in zip(heights, widths, strict=True)
            ]
        )
        return latents

    # MARK: encode_image
    @torch.inference_mode()
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
        latents = (
            encode_output[0].sample() - self.vae.shift_factor
        ) * self.vae.scaling_factor

        return latents

    @torch.no_grad()
    def decode_image(
        self,
        latents: torch.Tensor,
    ) -> list[Image.Image]:
        image = self.vae.decode(
            latents / self.vae.scaling_factor + self.vae.shift_factor,  # type: ignore
            return_dict=False,
        )[0]
        image = tensor_utils.tensor_to_images(image)

        return image

    @torch.no_grad()
    def decode_nested_image(
        self,
        latents: torch.Tensor,  # nested tensor
    ) -> list[Image.Image]:
        images = []
        for latent in latents:
            image = self.decode_image(latent.unsqueeze(0))
            images.extend(image)

        return images

    def prepare_timesteps(
        self, num_inference_steps: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self.scheduler.get_timesteps(num_inference_steps)
        sigmas = self.scheduler.get_sigmas(timesteps)

        return (
            torch.from_numpy(timesteps).to(device),
            torch.from_numpy(sigmas).to(device),
        )

    def prepare_encoder_hidden_states(
        self,
        encoder_output: TextEncodingOutput,
        do_cfg: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if do_cfg:
            prompt_embeddings = torch.cat(
                [
                    encoder_output.positive_embeddings,
                    encoder_output.negative_embeddings,
                ],
                dim=0,
            )
            mask = torch.cat(
                [
                    encoder_output.positive_attention_mask,
                    encoder_output.negative_attention_mask,
                ],
                dim=0,
            )
        else:
            prompt_embeddings = encoder_output.positive_embeddings
            mask = encoder_output.positive_attention_mask

        return prompt_embeddings, mask

    def _validate_batch_inputs(
        self,
        prompts: str | list[str],
        negative_prompts: str | list[str] | None,
        width: int | list[int],
        height: int | list[int],
    ) -> tuple[list[str], list[str] | None, list[int], list[int]]:
        if isinstance(prompts, str):
            prompts = [prompts]
        assert len(prompts) > 0, "At least one prompt is required."

        # negative prompts
        if negative_prompts is not None:
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts]
            if len(negative_prompts) == 1:
                negative_prompts = negative_prompts * len(prompts)

        # width
        if isinstance(width, int):
            width = [width]
        if len(width) == 1:
            width = width * len(prompts)

        # height
        if isinstance(height, int):
            height = [height]
        if len(height) == 1:
            height = height * len(prompts)

        assert len(prompts) == len(width) == len(height), (
            "The length of prompts, width, and height must be the same. But got "
            f"{len(prompts)}, {len(width)}, and {len(height)}."
        )

        return prompts, negative_prompts, width, height

    def chunk_cfg_velocity(
        self,
        velocity: torch.Tensor,  # nested tensor
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = velocity.unbind(dim=0)

        return (
            torch.nested.as_nested_tensor(values[:batch_size]),
            torch.nested.as_nested_tensor(values[batch_size:]),
        )

    # MARK: generate
    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int | list[int] = 768,
        height: int | list[int] = 768,
        num_inference_steps: int = 25,
        cfg_scale: float = 5.0,
        max_token_length: int = 256,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device("cuda") if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
        timesteps, sigmas = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            device=execution_device,
        )
        prompts, negative_prompts, widths, heights = self._validate_batch_inputs(
            prompts=prompt,
            negative_prompts=negative_prompt,
            width=width,
            height=height,
        )
        batch_size = len(prompts)

        # 2. Encode text
        encoder_output = self.text_encoder.encode_prompts(
            prompts=prompts,
            negative_prompts=negative_prompts,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )

        # 3. Prepare latents
        latents = self.prepare_nested_latents(
            heights=heights,
            widths=widths,
            dtype=execution_dtype,
            device=execution_device,
            seed=seed,
        )
        prompt_embeddings, prompt_mask = self.prepare_encoder_hidden_states(
            encoder_output=encoder_output,
            do_cfg=do_cfg,
        )

        # 4. Denoise
        # prepare refined prompt feature cache
        prompt_feature_cache: torch.Tensor | None = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000.0 -> 1.0
            for i, current_timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latents_input = torch.cat([latents] * 2) if do_cfg else latents

                timestep = current_timestep.repeat(batch_size).to(execution_device)

                # predict velocity and cache prompt features
                velocity_pred, prompt_mask, prompt_feature_cache = self.denoiser(
                    latents=latents_input,
                    caption_features=prompt_embeddings,
                    timestep=timestep,
                    caption_mask=prompt_mask,
                    cached_caption_features=prompt_feature_cache,
                )

                # perform cfg
                if do_cfg:
                    velocity_pred_positive, velocity_pred_negative = (
                        self.chunk_cfg_velocity(velocity_pred, batch_size)
                    )
                    velocity_pred = velocity_pred_negative + cfg_scale * (
                        velocity_pred_positive - velocity_pred_negative
                    )

                # denoise the latents
                current_sigma, next_sigma = sigmas[i], sigmas[i + 1]
                latents = self.scheduler.step(
                    latent=latents,
                    velocity_pred=velocity_pred,
                    sigma=current_sigma,
                    next_sigma=next_sigma,
                )

                progress_bar.update(1)

        # 5. Decode latents
        images = self.decode_nested_image(latents)

        return images
