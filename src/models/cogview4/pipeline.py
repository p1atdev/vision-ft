from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file

from .denoiser import Denoiser
from .vae import VAE
from .text_encoder import TextEncoder, DEFAULT_MAX_TOKEN_LENGTH
from .config import CogView4Config
from .scheduler import calculate_time_shift

from ...modules.quant import replace_by_prequantized_weights
from ...modules.timestep.scheduler import get_linear_schedule
from ...modules.timestep.sampling import time_shift_linear
from ...utils import tensor as tensor_utils


def convert_from_original_key(key: str) -> str:
    key = key.replace("diffusion_model.", "denoiser.", 1)
    key = key.replace("text_encoder.", "text_encoder.model.", 1)

    return key


def convert_to_original_key(key: str) -> str:
    key = key.replace("denoiser.", "diffusion_model.", 1)
    key = key.replace("text_encoder.model.", "text_encoder.", 1)

    return key


class CogView4ModelForInference(nn.Module):
    denoiser: Denoiser
    denoiser_class: type[Denoiser] = Denoiser

    vae: VAE
    text_encoder: TextEncoder

    def __init__(self, config: CogView4Config):
        super().__init__()

        self.config = config

        self.denoiser = self.denoiser_class(config.denoiser)
        vae = VAE.from_default()
        assert isinstance(vae, VAE)
        self.vae = vae
        self.text_encoder = TextEncoder.from_default()

        self.progress_bar = tqdm

    @classmethod
    def from_config(cls, config: CogView4Config) -> "CogView4ModelForInference":
        return cls(config)

    def _from_checkpoint(
        self,
        strict: bool = True,
    ):
        config = self.config
        state_dict = load_file(config.checkpoint_path)
        # TODO: key conversion
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
        config: CogView4Config,
    ) -> "CogView4ModelForInference":
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

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        height: int,
        width: int,
        device: torch.device,
    ):
        image_seq_len = (
            (height // self.vae.compression_ratio)
            * (width // self.vae.compression_ratio)
            // (self.denoiser.patch_size**2)
        )
        timesteps = (
            get_linear_schedule(
                num_inference_steps,
                device,
                start=1000.0,
                end=1.0,
            )
            .to(torch.int64)
            .to(torch.float32)
        )
        sigmas = timesteps / 1000.0
        mu = calculate_time_shift(image_seq_len)
        sigmas = time_shift_linear(mu, t=sigmas)
        sigmas = torch.cat(
            [sigmas, torch.zeros(1, device=device)]  # avoid out of index error
        )

        return timesteps, sigmas

    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 768,
        height: int = 768,
        original_size: tuple[int, int] | None = None,
        target_size: tuple[int, int] | None = None,
        crop_coords_top_left: tuple[int, int] = (0, 0),
        num_inference_steps: int = 20,
        cfg_scale: float = 3.5,
        seed: int | None = None,
        max_token_length: int = DEFAULT_MAX_TOKEN_LENGTH,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        denoiser_dtype = next(self.denoiser.parameters()).dtype
        do_cfg = cfg_scale > 1.0
        timesteps, sigmas = self.prepare_timesteps(
            num_inference_steps, height=height, width=width, device=execution_device
        )
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        original_size = original_size or (height, width)
        original_size_tensor = torch.tensor(
            original_size, device=execution_device
        ).repeat(batch_size, 1)
        target_size = target_size or (height, width)
        target_size_tensor = torch.tensor(target_size, device=execution_device).repeat(
            batch_size, 1
        )
        crop_coords_tensor = torch.tensor(
            crop_coords_top_left, device=execution_device
        ).repeat(batch_size, 1)

        # 2. Encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )
        if do_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 3. Prepare latents.
        latents = self.prepare_latents(
            batch_size,
            height,
            width,
            denoiser_dtype,
            execution_device,
            seed=seed,
        )

        # 4. Denoising loop
        if do_offloading and self.denoiser.offload_strategy is None:
            self.denoiser.to(execution_device)
        latents = latents.to(execution_device)
        prompt_embeddings = torch.cat(
            [
                encoder_output.positive_embeddings,
                encoder_output.negative_embeddings,
            ]
        ).to(execution_device)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000.0 -> 1.0
            for i, current_timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents

                # aura use timestep value between 0 and 1, with t=1 as noise and t=0 as the image
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                batch_timestep = current_timestep.expand(
                    latent_model_input.shape[0]
                ).to(latents.device, dtype=latents.dtype)

                # predict noise model_output
                velocity_pred = self.denoiser(
                    latent=latent_model_input,
                    encoder_hidden_states=prompt_embeddings,
                    timestep=batch_timestep,
                    original_size=original_size_tensor,
                    target_size=target_size_tensor,
                    crop_coords=crop_coords_tensor,
                )

                # perform cfg
                if do_cfg:
                    velocity_pred_positive, velocity_pred_negative = (
                        velocity_pred.chunk(2)
                    )
                    velocity_pred = velocity_pred_negative + cfg_scale * (
                        velocity_pred_positive - velocity_pred_negative
                    )

                # denoise the latents
                next_sigma = sigmas[i + 1]
                current_sigma = sigmas[i]
                latents = latents + velocity_pred * (next_sigma - current_sigma)

                progress_bar.update()

        if do_offloading:
            self.denoiser.to("cpu")
            torch.cuda.empty_cache()

        # 5. Decode the latents
        if do_offloading:
            self.vae.to(execution_device)  # type: ignore
        image = self.decode_image(latents.to(self.vae.device))
        if do_offloading:
            self.vae.to("cpu")  # type: ignore
            torch.cuda.empty_cache()

        return image
