from PIL import Image
import torch

from ....modules.loss.flow_match import (
    convert_x0_to_velocity,
    ModelPredictionType,
)
from ....modules.timestep.scheduler import get_linear_schedule


from ..config import SDXLConfig
from ..pipeline import SDXLModel


class SDXLFlowMatchConfig(SDXLConfig):
    model_prediction: ModelPredictionType = "velocity"
    noise_scale: float = 1.0


class SDXLFlowMatch(SDXLModel):
    config: SDXLFlowMatchConfig

    def __init__(self, config: SDXLFlowMatchConfig):
        super().__init__(config)

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
    ):
        timesteps = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device)

        return timesteps

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
        max_token_length: int = 75,
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
        timesteps = self.prepare_timesteps(
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
        if do_offloading:
            self.denoiser.to(execution_device)
        latents = (
            self.prepare_latents(
                batch_size,
                height,
                width,
                execution_dtype,
                execution_device,
                max_noise_sigma=1.0,
                seed=seed,
            )
            * self.config.noise_scale
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
            for i, current_timestep in enumerate(timesteps[:-1]):
                # expand latents if doing cfg
                latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
                batch_timestep = current_timestep.expand(latent_model_input.size(0)).to(
                    execution_device
                )

                # predict noise model_output
                model_pred = self.denoiser(
                    latents=latent_model_input,
                    timestep=batch_timestep,
                    encoder_hidden_states=prompt_embeddings,
                    encoder_pooler_output=pooled_prompt_embeddings,
                    original_size=original_size_tensor,
                    target_size=target_size_tensor,
                    crop_coords_top_left=crop_coords_tensor,
                )
                if self.config.model_prediction == "image":
                    # convert x0 to velocity
                    velocity_pred = convert_x0_to_velocity(
                        x0=model_pred,
                        noisy_latents=latent_model_input,
                        timestep=batch_timestep,
                    )
                elif self.config.model_prediction == "velocity":
                    velocity_pred = model_pred
                else:
                    # TODO: support noise?

                    raise ValueError(
                        f"Unknown model_prediction: {self.config.model_prediction}"
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
                latents = latents + velocity_pred * (timesteps[i + 1] - timesteps[i])

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
