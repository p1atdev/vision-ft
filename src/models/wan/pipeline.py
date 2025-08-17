from tqdm import tqdm
from PIL import Image

import numpy as np
import torch
import torch.nn as nn

from safetensors.torch import load_file
from accelerate import init_empty_weights

from .config import WanConfig
from .denoiser import (
    Denoiser,
)
from .text_encoder import TextEncoder, TextEncodingOutput
from .vae import VAE

# from .scheduler import Scheduler
from .util import convert_from_original_key
from ...utils import tensor as tensor_utils
from ...modules.quant import replace_by_prequantized_weights


class Wan22(nn.Module):
    denoiser: Denoiser
    denoiser_class: type[Denoiser] = Denoiser

    text_encoder: TextEncoder
    text_encoder_class: type[TextEncoder] = TextEncoder

    vae: VAE
    vae_class: type[VAE] = VAE

    def __init__(self, config: WanConfig):
        super().__init__()

        self.config = config

        self._setup_models(config)

    def _setup_models(self, config):
        self.denoiser = self.denoiser_class(config.denoiser)
        self.vae = VAE.from_default()
        self.text_encoder = TextEncoder.from_default()

        # self.scheduler = Scheduler()  # euler

        self.progress_bar = tqdm

    @classmethod
    def from_config(cls, config: WanConfig) -> "Wan22":
        return cls(config)

    def _from_checkpoint(
        self,
        strict: bool = True,
    ):
        config = self.config

        # Denoiser
        denoiser_state_dict = load_file(config.denoiser_path)
        # prepare for prequantized weights
        replace_by_prequantized_weights(self, denoiser_state_dict)
        self.denoiser.load_state_dict(
            {
                convert_from_original_key(key, "denoiser"): value
                for key, value in denoiser_state_dict.items()
            },
            strict=strict,
            assign=True,
        )

        # VAE
        vae_state_dict = load_file(config.vae_path)
        replace_by_prequantized_weights(self, vae_state_dict)
        self.vae.load_state_dict(
            {
                convert_from_original_key(key, "vae"): value
                for key, value in vae_state_dict.items()
            },
            strict=strict,
            assign=True,
        )

        # Text Encoder
        text_encoder_state_dict = load_file(config.text_encoder_path)
        replace_by_prequantized_weights(self, text_encoder_state_dict)
        self.text_encoder.load_state_dict(
            {
                convert_from_original_key(key, "text_encoder"): value
                for key, value in text_encoder_state_dict.items()
            },
            strict=strict,
            assign=True,
        )

    @classmethod
    def from_checkpoint(
        cls,
        config: WanConfig,
    ) -> "Wan22":
        with init_empty_weights():
            model = cls.from_config(config)

        model._from_checkpoint()

        return model

    def prepare_latents(
        self,
        batch_size: int,
        frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None = None,
    ):
        latent_channels = self.denoiser.config.in_channels

        shape = (
            batch_size,
            latent_channels,
            frames // self.vae.temporal_compression_ratio,
            height // self.vae.spatial_compression_ratio,
            width // self.vae.spatial_compression_ratio,
        )

        latents = tensor_utils.incremental_seed_randn(
            shape=shape,
            seed=seed,
            dtype=dtype,
            device=device,
        )

        return latents

    @torch.inference_mode()
    def encode_video(
        self,
        video: Image.Image | list[Image.Image] | list[list[Image.Image]] | torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(video, Image.Image):
            video_list: list[list[Image.Image]] = [[video]]
        elif isinstance(video, list) and isinstance(video[0], Image.Image):
            video_list = [video]  # type: ignore
        elif (
            isinstance(video, list)
            and isinstance(video[0], list)
            and isinstance(video[0][0], Image.Image)
        ):
            video_list = video  # type: ignore
        else:
            # already tensor
            video_tensor = video  # type: ignore

        if not isinstance(video_list, torch.Tensor):
            video_tensor = tensor_utils.videos_to_tensor(
                video_list, self.vae.dtype, self.vae.device
            )

        encode_output = self.vae.encode(video_tensor.to(self.vae.dtype))  # type: ignore
        latents = (
            encode_output.latent_dist.sample() - self.vae.shift_factor  # type: ignore
        ) * self.vae.scaling_factor

        return latents

    @torch.inference_mode()
    def decode_video(
        self,
        latents: torch.Tensor,
    ) -> list[list[Image.Image]]:
        video_tensor = self.vae.decode(
            latents / self.vae.scaling_factor + self.vae.shift_factor,  # type: ignore
            return_dict=False,
        )[0]
        videos = tensor_utils.tensor_to_videos(video_tensor)

        return videos

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "Wan22 does not support prepare_timesteps, use Scheduler instead."
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
    ) -> tuple[list[str], list[str] | None]:
        if isinstance(prompts, str):
            prompts = [prompts]
        assert len(prompts) > 0, "At least one prompt is required."

        # negative prompts
        if negative_prompts is not None:
            if isinstance(negative_prompts, str):
                negative_prompts = [negative_prompts]
            if len(negative_prompts) == 1:
                negative_prompts = negative_prompts * len(prompts)

        return prompts, negative_prompts

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
        frames: int = 16,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 25,
        cfg_scale: float = 5.0,
        max_token_length: int = 512,
        seed: int | None = None,
        execution_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[list[Image.Image]]:
        # 1. Prepare args
        execution_device: torch.device = (
            torch.device("cuda") if isinstance(device, str) else device
        )
        do_cfg = cfg_scale > 1.0
        timesteps, sigmas = self.prepare_timesteps(
            num_inference_steps=num_inference_steps,
            device=execution_device,
        )
        prompts, negative_prompts = self._validate_batch_inputs(
            prompts=prompt,
            negative_prompts=negative_prompt,
        )
        batch_size = len(prompts)

        # 2. Encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompts=prompts,
            negative_prompts=negative_prompts,
            use_negative_prompts=do_cfg,
            max_token_length=max_token_length,
        )
        if do_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 3. Prepare latents
        latents = self.prepare_latents(
            batch_size=batch_size,
            frames=frames,
            height=height,
            width=width,
            dtype=execution_dtype,
            device=execution_device,
            seed=seed,
        )

        # 4. Denoise
        if do_offloading:
            self.denoiser.to(execution_device)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1.0 -> 0.0
            for i, current_timestep in enumerate(timesteps):
                # 1.0 -> 0.0
                timestep_input = current_timestep.repeat(batch_size).to(
                    execution_device
                )

                if do_cfg:
                    # expand the latents if we are doing classifier free guidance
                    latents_input = torch.cat([latents] * 2)
                    timestep_input = torch.cat([timestep_input] * 2)
                else:
                    latents_input = latents

                # prepare the prompt features
                prompt_embeddings, prompt_mask = self.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=do_cfg,
                )

                # predict velocity
                velocity_pred = self.denoiser(
                    latents=latents_input,
                    context=prompt_embeddings,
                    timesteps=timestep_input,
                )

                # perform cfg
                if do_cfg:
                    velocity_pred_positive, velocity_pred_negative = (
                        self.chunk_cfg_velocity(
                            velocity_pred,
                            batch_size,
                        )
                    )
                    velocity_pred = (
                        velocity_pred_negative
                        + (velocity_pred_positive - velocity_pred_negative) * cfg_scale
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

        if do_offloading:
            self.denoiser.to("cpu")
            torch.cuda.empty_cache()

            self.vae.to(execution_device)  # type: ignore

        # 5. Decode latents
        videos = self.decode_video(latents)
        if do_offloading:
            self.vae.to("cpu")  # type: ignore
            torch.cuda.empty_cache()

        return videos
