from tqdm import tqdm
from PIL import Image
import warnings

import torch
from torch._tensor import Tensor
import torch.nn as nn

from transformers import AutoTokenizer
from safetensors.torch import load_file
from accelerate import init_empty_weights

from .config import FluxConfig
from .denoiser import Denoiser, DENOISER_TENSOR_PREFIX
from .text_encoder import (
    TextEncoder,
    TEXT_ENCODER_T5_TENSOR_PREFIX,
    TEXT_ENCODER_CLIP_TENSOR_PREFIX,
)
from .vae import (
    VAE,
    VAE_TENSOR_PREFIX,
    DEFAULT_VAE_CONFIG,
    detect_vae_type,
    DEFAULT_VAE_FOLDER,
)
from ...utils import tensor as tensor_utils
from ...modules.quant import replace_by_prequantized_weights
from ...modules.timestep.scheduler import get_flux_schedule, get_linear_schedule

FLUX1_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"


def convert_to_original_key(key: str) -> str:
    key = key.replace("denoiser.", DENOISER_TENSOR_PREFIX)
    key = key.replace("vae.", VAE_TENSOR_PREFIX)
    key = key.replace("text_encoder.clip.", TEXT_ENCODER_CLIP_TENSOR_PREFIX)
    key = key.replace("text_encoder.t5.", TEXT_ENCODER_T5_TENSOR_PREFIX)

    return key


def convert_to_comfy_key(key: str) -> str:
    key = key.replace("denoiser.", "diffusion_model.")
    key = key.replace("vae.", VAE_TENSOR_PREFIX)
    key = key.replace("text_encoder.clip.", TEXT_ENCODER_CLIP_TENSOR_PREFIX)
    key = key.replace("text_encoder.t5.", TEXT_ENCODER_T5_TENSOR_PREFIX)

    return key


def convert_from_original_key(key: str) -> str:
    key = key.replace("model.diffusion_model.", "denoiser.")
    key = key.replace("diffusion_model.", "denoiser.")
    key = key.replace(DENOISER_TENSOR_PREFIX, "denoiser.")
    key = key.replace(VAE_TENSOR_PREFIX, "vae.")
    key = key.replace(TEXT_ENCODER_CLIP_TENSOR_PREFIX, "text_encoder.clip.")
    key = key.replace(TEXT_ENCODER_T5_TENSOR_PREFIX, "text_encoder.t5.")

    return key


class FluxModel(nn.Module):
    denoiser: Denoiser
    denoiser_class: type[Denoiser] = Denoiser

    def __init__(self, config: FluxConfig):
        super().__init__()

        self.config = config

        self.denoiser = self.denoiser_class.from_config(config.denoiser)
        vae = VAE.from_config(DEFAULT_VAE_CONFIG)
        assert isinstance(vae, VAE)
        self.vae = vae
        self.text_encoder = TextEncoder.from_default()

        # self.scheduler = Scheduler()
        self.progress_bar = tqdm

    @classmethod
    def from_config(cls, config: FluxConfig) -> "FluxModel":
        return cls(config)

    def load_checkpoint_weights(
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
        vae_type = detect_vae_type(state_dict)
        if vae_type == "original":
            warnings.warn(
                "Original VAE weights are not supported. Using the default VAE weights."
            )
            self.vae = VAE.from_pretrained(
                FLUX1_SCHNELL_REPO,
                subfolder=DEFAULT_VAE_FOLDER,
                torch_dtype=torch.bfloat16,
            )
        elif vae_type == "autoencoder_kl":
            self.vae.load_state_dict(  # type: ignore
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

    # replace the prefix to match the original checkpoint
    def state_dict(
        self,
        destination: dict[str, torch.Tensor] | None = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> dict[str, torch.Tensor]:
        state_dict: dict[str, torch.Tensor] = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state_dict = {
            tensor_utils.remove_orig_mod_prefix(key): value
            for key, value in state_dict.items()
        }

        # shared.weight is referenced by its encoder,
        # and removed when saving with safetensors' save_model().
        # We need to de-reference it here.
        state_dict["text_encoder.t5.shared.weight"] = state_dict[
            "text_encoder.t5.shared.weight"
        ].clone()

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
        latent_channels = self.vae.config.latent_channels  # type: ignore

        if latents is None:
            shape = (
                batch_size,
                latent_channels,
                int(height) // self.vae.compression_ratio,  # type: ignore
                int(width) // self.vae.compression_ratio,  # type: ignore
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
                image_list,
                self.vae.dtype,  # type: ignore
                self.vae.device,  # type: ignore
            )
        else:
            image_tensor = image
        encode_output = self.vae.encode(image_tensor.to(self.vae.dtype))  # type: ignore
        latents = encode_output[0].sample() * self.vae.scaling_factor  # type: ignore

        return latents

    @torch.no_grad()
    def decode_image(
        self,
        latents: torch.Tensor,
    ) -> list[Image.Image]:
        image = self.vae.decode(  # type: ignore
            latents / self.vae.scaling_factor,  # type: ignore
            return_dict=False,
        )[0]
        image = tensor_utils.tensor_to_images(image)

        return image

    def generate(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        width: int = 768,
        height: int = 768,
        num_inference_steps: int = 20,
        cfg_scale: float = 1.0,
        distilled_guidance_scale: float = 1.0,
        seed: int | None = None,
        max_token_length: int = 512,
        device: torch.device | str = torch.device("cuda"),
        do_offloading: bool = False,
    ) -> list[Image.Image]:
        # 1. Prepare args
        execution_device = device
        denoiser_dtype = torch.bfloat16
        do_cfg = cfg_scale > 1.0
        device = torch.device(device) if isinstance(device, str) else device
        assert isinstance(device, torch.device)

        batch_size = len(prompt) if isinstance(prompt, list) else 1

        # 2. Encode text
        if do_offloading:
            self.text_encoder.to(device)
        encoder_output = self.text_encoder.encode_prompts(
            prompt,
            negative_prompt,
            use_negative_prompts=do_cfg,
            t5_max_token_length=max_token_length,
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
            device=execution_device,  # type: ignore
            seed=seed,
        )

        # 3.5 Prepare timesteps
        # TODO: scheduler をモジュールにして透過的にする
        # dev is not supported
        timesteps = get_linear_schedule(num_inference_steps, execution_device=device)
        delta_timestep = 1.0 / num_inference_steps

        # 4. Denoising loop
        if do_offloading:
            self.denoiser.to(device)
        latents = latents.to(self.denoiser.device)
        t5_prompt_embeddings = torch.cat(
            [
                encoder_output.t5.positive_embeddings,
                encoder_output.t5.negative_embeddings,
            ]
        ).to(self.denoiser.device)
        clip_prompt_embeddings = torch.cat(
            [
                encoder_output.clip.positive_embeddings,
                encoder_output.clip.negative_embeddings,
            ]
        ).to(self.denoiser.device)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # current_timestep is 1000.0 -> 0
            for i, current_timestep in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                batch_timestep = torch.tensor([current_timestep]).expand(
                    latent_model_input.shape[0]
                )
                batch_timestep = batch_timestep.to(latents.device, dtype=latents.dtype)
                batch_distilled_guidance_scale = torch.full_like(
                    batch_timestep, distilled_guidance_scale
                )

                # predict noise model_output
                velocity_pred = self.denoiser.forward(
                    latent=latent_model_input,
                    t5_hidden_states=t5_prompt_embeddings,
                    timesteps=batch_timestep,
                    clip_hidden_states=clip_prompt_embeddings,
                    guidance=batch_distilled_guidance_scale,
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
                latents -= velocity_pred * delta_timestep

                progress_bar.update()

        if do_offloading:
            self.denoiser.to("cpu")
            torch.cuda.empty_cache()

        # 5. Decode the latents
        if do_offloading:
            self.vae.to(device)  # type: ignore
        image = self.decode_image(latents.to(self.vae.device))  # type: ignore
        if do_offloading:
            self.vae.to("cpu")  # type: ignore
            torch.cuda.empty_cache()

        return image
