import click

import torch

from accelerate import init_empty_weights

from src.models.sdxl.adapter.flow_match import SDXLFlowMatch, SDXLFlowMatchConfig
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_scaled_noised_latents,
    loss_with_predicted_velocity,
    loss_with_predicted_image,
)
from src.modules.timestep.sampling import TimestepSamplingType, sample_timestep

from text_to_image import SDXLForTextToImageTraining


class SDXLForFlowMatchingTrainingConfig(SDXLFlowMatchConfig):
    max_token_length: int = 225  # 75 * 3

    timestep_sampling: TimestepSamplingType = "shift_sigmoid"
    timestep_std: float = 0.8
    timestep_mean: float = -0.8


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
            timesteps = (
                sample_timestep(
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
                * 1000  # 0~1000
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_scaled_noised_latents(
            latents=latents,
            timestep=timesteps / 1000,  # 0.0~1.0
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
            timestep=timesteps / 1000,  # 0.0~1.0
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
