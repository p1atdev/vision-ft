from PIL import Image
import click

import torch

from accelerate import init_empty_weights

from src.models.sdxl import SDXLModel
from src.models.sdxl.config import SDXLFlowMatchConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_scaled_noised_latents,
    loss_with_predicted_velocity,
    loss_with_predicted_image,
    convert_x0_to_velocity,
    ModelPredictionType,
)
from src.modules.timestep.scheduler import get_linear_schedule
from src.modules.timestep.sampling import TimestepSamplingType, sample_timestep

from text_to_image import SDXLForTextToImageTraining


class SDXLForFlowMatchingTrainingConfig(SDXLFlowMatchConfig):
    max_token_length: int = 225  # 75 * 3

    model_prediction: ModelPredictionType = "velocity"
    # loss always uses velocity

    noise_scale: float = 1.0

    timestep_sampling: TimestepSamplingType = "shift_sigmoid"
    timestep_std: float = 0.8
    timestep_mean: float = -0.8


class SDXLFlowMatch(SDXLModel):
    config: SDXLFlowMatchConfig

    def __init__(self, config: SDXLFlowMatchConfig):
        super().__init__(config)

    def prepare_timesteps(
        self,
        num_inference_steps: int,
        device: torch.device,
    ):
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
        sigmas = torch.cat(
            [sigmas, torch.zeros(1, device=device)]  # avoid out of index error
        )

        return timesteps, sigmas

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
            for i, current_timestep in enumerate(timesteps):
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
                if self.config.model_prediction == "real_image":
                    # convert x0 to velocity
                    model_pred = convert_x0_to_velocity(
                        x0=model_pred,
                        noisy_latents=latent_model_input,
                        timestep=batch_timestep,
                    )
                velocity_pred = model_pred

                # perform cfg
                if do_cfg:
                    velocity_pred_positive, velocity_pred_negative = (
                        velocity_pred.chunk(2)
                    )
                    velocity_pred = velocity_pred_negative + cfg_scale * (
                        velocity_pred_positive - velocity_pred_negative
                    )

                # denoise the latents
                current_sigma, next_sigma = sigmas[i], sigmas[i + 1]
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


class SDXLForFlowMatchingTraining(SDXLForTextToImageTraining):
    model: SDXLFlowMatch

    model_config: SDXLForFlowMatchingTrainingConfig
    model_config_class = SDXLForFlowMatchingTrainingConfig

    def setup_model(self):
        with init_empty_weights():
            self.model = SDXLFlowMatch(self.model_config)

            # freeze other modules
            self.model.text_encoder.eval()
            self.model.vae.eval()  # type: ignore

        self.model._from_checkpoint()  # load!

    def treat_loss(
        self,
        model_pred: torch.Tensor,
        latents: torch.Tensor,
        random_noise: torch.Tensor,
        noisy_latents: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        if self.model_config.model_prediction == "velocity":
            return loss_with_predicted_velocity(
                latents=latents,
                random_noise=random_noise,
                predicted_velocity=model_pred,
            )
        elif self.model_config.model_prediction == "image":
            return loss_with_predicted_image(
                latents=latents,
                noisy_latents=noisy_latents,
                timestep=timestep,
                predicted_image=model_pred,
            )

        else:
            raise ValueError(
                f"Unknown model_prediction: {self.model_config.model_prediction}"
            )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]
        original_size = batch["original_size"]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_output = self.model.text_encoder.encode_prompts(caption)
            encoder_hidden_states, pooled_hidden_states = (
                self.model.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=False,
                    device=self.accelerator.device,
                )
            )

            latents = self.model.encode_image(pixel_values)
            timesteps = sample_timestep(
                latents_shape=latents.shape,
                device=self.accelerator.device,
                # jit sampling
                std=self.model_config.timestep_std,
                mean=self.model_config.timestep_mean,
                # if we use flux shifted sigmoid:
                sampling_type=self.model_config.timestep_sampling,
                shift=3.1825,
                sigmoid_scale=1,
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_scaled_noised_latents(
            latents=latents,
            timestep=timesteps,
            noise_scale=self.model_config.noise_scale,
        )

        # 3. Predict the noise
        model_pred = self.model(
            latents=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_pooler_output=pooled_hidden_states,
            original_size=original_size,
            target_size=target_size,
            crop_coords_top_left=crop_coords_top_left,
        )

        # 4. Calculate the loss
        l2_loss = self.treat_loss(
            model_pred=model_pred,
            latents=latents,
            random_noise=random_noise,
            noisy_latents=noisy_latents,
            timestep=timesteps,
        )
        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(SDXLForFlowMatchingTraining)

    trainer.train()


if __name__ == "__main__":
    main()
