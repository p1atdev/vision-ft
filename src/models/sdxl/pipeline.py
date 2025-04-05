from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn

from accelerate import init_empty_weights
from safetensors.torch import load_file

from .denoiser import Denoiser
from .vae import VAE
from .text_encoder import TextEncoder, MultipleTextEncodingOutput
from .config import SDXLConfig
from .scheduler import Scheduler

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

        self.scheduler = Scheduler()  # euler discrete

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

    # MARK: prepare_latents
    def prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        max_noise_sigma: float | torch.Tensor,
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
            latents = (
                tensor_utils.incremental_seed_randn(
                    shape,
                    seed=seed,
                    dtype=dtype,
                    device=device,
                )
                * max_noise_sigma
            )
        else:
            latents = latents.to(dtype=dtype, device=device)

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
        latents = encode_output[0].sample() * self.vae.scaling_factor

        return latents

    # MARK: decode_image
    @torch.inference_mode()
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
        encoder_output: MultipleTextEncodingOutput,
        do_cfg: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if do_cfg:
            prompt_embeddings = torch.cat(
                [
                    torch.cat(
                        [
                            encoder_output.text_encoder_1.positive_embeddings,
                            encoder_output.text_encoder_2.positive_embeddings,
                        ],
                        dim=-1,
                    ),  # (batch_size, 77, 1280) + (batch_size, 77, 768) -> (batch_size, 77, 2048)
                    torch.cat(
                        [
                            encoder_output.text_encoder_1.negative_embeddings,
                            encoder_output.text_encoder_2.negative_embeddings,
                        ],
                        dim=-1,
                    ),  # (batch_size, 77, 1280) + (batch_size, 77, 768) -> (batch_size, 77, 2048)
                ],
                dim=0,  # concat positive and negative embeddings in batch dim
            ).to(device)
            pooled_prompt_embeddings = torch.cat(
                [
                    encoder_output.text_encoder_2.pooled_positive_embeddings,
                    encoder_output.text_encoder_2.pooled_negative_embeddings,
                ],
                dim=0,
            ).to(device)
        else:
            prompt_embeddings = torch.cat(
                [
                    encoder_output.text_encoder_1.positive_embeddings,
                    encoder_output.text_encoder_2.positive_embeddings,
                ],
                dim=-1,
            ).to(device)
            pooled_prompt_embeddings = (
                encoder_output.text_encoder_2.pooled_positive_embeddings.to(device)
            )

        return prompt_embeddings, pooled_prompt_embeddings

    # MARK: generate
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
        batch_size = len(prompt) if isinstance(prompt, list) else 1
        original_size = original_size or (height, width)
        original_size_tensor = torch.tensor(original_size, device=execution_device)
        target_size = target_size or (height, width)
        target_size_tensor = torch.tensor(target_size, device=execution_device)
        crop_coords_tensor = torch.tensor(crop_coords_top_left, device=execution_device)

        # 2. Encode text
        if do_offloading:
            self.text_encoder.to(execution_device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
        )
        if do_offloading:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()

        # 3. Prepare latents, etc.
        if do_offloading and self.denoiser.offload_strategy is None:
            self.denoiser.to(execution_device)
        latents = self.prepare_latents(
            batch_size,
            height,
            width,
            execution_dtype,
            execution_device,
            max_noise_sigma=self.scheduler.get_max_noise_sigma(sigmas),
            seed=seed,
        )
        prompt_embeddings, pooled_prompt_embeddings = (
            self.prepare_encoder_hidden_states(
                encoder_output=encoder_output,
                do_cfg=do_cfg,
                device=execution_device,
            )
        )
        original_size_tensor = original_size_tensor.expand(
            prompt_embeddings.size(0), -1
        )
        target_size_tensor = target_size_tensor.expand(prompt_embeddings.size(0), -1)
        crop_coords_tensor = crop_coords_tensor.expand(prompt_embeddings.size(0), -1)

        # 4. Denoise
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000 -> 1
            for i, current_timestep in enumerate(timesteps):
                current_sigma, next_sigma = sigmas[i], sigmas[i + 1]

                # expand latents if doing cfg
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, current_sigma
                )

                batch_timestep = current_timestep.expand(latent_model_input.size(0)).to(
                    execution_device
                )

                # predict noise model_output
                noise_pred = self.denoiser(
                    latents=latent_model_input,
                    timestep=batch_timestep,
                    encoder_hidden_states=prompt_embeddings,
                    encoder_pooler_output=pooled_prompt_embeddings,
                    original_size=original_size_tensor,
                    target_size=target_size_tensor,
                    crop_coords_top_left=crop_coords_tensor,
                )

                # perform cfg
                if do_cfg:
                    noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                    noise_pred = noise_pred_negative + cfg_scale * (
                        noise_pred_positive - noise_pred_negative
                    )

                # denoise the latents
                latents = latents + noise_pred * (next_sigma - current_sigma)

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

    def forward(self, *args, **kwargs):
        return self.denoiser(*args, **kwargs)
