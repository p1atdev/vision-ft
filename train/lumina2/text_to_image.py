from PIL.Image import Image
from typing import Literal
import click

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nested import to_padded_tensor

from accelerate import init_empty_weights

from src.models.lumina2 import Lumina2Config, Lumina2, convert_to_comfy_key
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.flow_match import (
    prepare_noised_latents,
    loss_with_predicted_velocity,
)
from src.modules.timestep.sampling import uniform_rand, shift_fraction_uniform_rand
from src.modules.peft import get_adapter_parameters
from src.utils.logging import wandb_image

torch._dynamo.config.capture_scalar_outputs = True  # type: ignore


class Lumina2ForTextToImageTrainingConfig(Lumina2Config):
    max_token_length: int = 256

    timestep_sampling: Literal[
        "uniform",
        "lognorm",
        "shift_fraction_uniform",
    ] = "uniform"
    timestep_fraction_divisible: list[int] = [20, 25, 30, 32]

    use_lowres_loss: bool = True
    use_downsampled_velocity_loss: bool = False


class Lumina2ForTextToImageTraining(ModelForTraining, nn.Module):
    model: Lumina2

    model_config: Lumina2ForTextToImageTrainingConfig
    model_config_class = Lumina2ForTextToImageTrainingConfig

    def setup_model(self):
        with init_empty_weights():
            self.model = Lumina2(self.model_config)

            # freeze other modules
            self.model.text_encoder.eval()
            self.model.vae.eval()  # type: ignore

        self.model._from_checkpoint()  # load!

    @property
    def raw_model(self) -> Lumina2:
        return self.accelerator.unwrap_model(self.model)

    def sanity_check(self):
        latent = self.model.prepare_nested_latents(
            heights=[96],
            widths=[96],
            dtype=torch.bfloat16,
            device=self.accelerator.device,
        )
        encoder_hidden_states = torch.randn(
            1,
            256,  # max token len
            self.model_config.denoiser.caption_dim,  # 2304
            device=self.accelerator.device,
        )
        caption_mask = torch.ones(
            1,
            256,  # max token len
            dtype=torch.bool,
            device=self.accelerator.device,
        )
        timestep = torch.tensor([0.1], device=self.accelerator.device)

        with self.accelerator.autocast():
            _velocity_pred, _, _ = self.model.denoiser(
                latents=latent,
                timestep=timestep,
                caption_features=encoder_hidden_states,
                caption_mask=caption_mask,
            )

    # for auxiliary loss
    def downsample_4x(self, latents: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            latents,
            kernel_size=4,
            stride=4,
        )

    def forward_and_loss(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        caption_features: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timestep,
        )

        # 3. Predict the noise
        velocity_pred, _, _ = model(
            latents=noisy_latents,
            timestep=timestep,
            caption_features=caption_features,
            caption_mask=caption_mask,
        )
        # actually no padding, just convert to normal tensor
        velocity_pred = -to_padded_tensor(velocity_pred, padding=0.0)

        # Lumina2's training target:
        # loss := rmse(prediction, clean_latents - random_noise)
        #       = rmse(-prediction, random_noise - clean_latents)
        loss = loss_with_predicted_velocity(
            latents=latents,
            random_noise=random_noise,
            predicted_velocity=velocity_pred,
        )

        return loss, velocity_pred, (random_noise - latents)

    def sample_timesteps(
        self,
        latents_shape: torch.Size,
    ) -> torch.Tensor:
        if self.model_config.timestep_sampling == "uniform":
            return uniform_rand(
                latents_shape=latents_shape,
                device=self.accelerator.device,
            )
        elif self.model_config.timestep_sampling == "lognorm":
            return self.model.scheduler.sample_sigmoid_randn(
                latents_shape=latents_shape,
                device=self.accelerator.device,
                patch_size=self.model.denoiser.patch_size,
            )
        elif self.model_config.timestep_sampling == "shift_fraction_uniform":
            # Lumina2 uses 0.0 -> 1.0 timesteps, so subtract from 1.0
            return 1 - shift_fraction_uniform_rand(
                latents_shape=latents_shape,
                device=self.accelerator.device,
                shift=self.model.scheduler.shift,
                divisible=self.model_config.timestep_fraction_divisible,
            )

        else:
            raise ValueError(
                f"Unknown timestep sampling method: {self.model_config.timestep_sampling}. "
            )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_output = self.model.text_encoder.encode_prompts(
                caption,
                max_token_length=self.model_config.max_token_length,
            )
            encoder_hidden_states, caption_mask = (
                self.model.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=False,
                )
            )

            latents = self.model.encode_image(pixel_values)
            # 0.0 ~ 1.0
            timesteps = self.sample_timesteps(
                latents_shape=latents.shape,
            )

        # 2~4. Predict and calculate the loss
        l2_highres_loss, highres_velocity, highres_target = self.forward_and_loss(
            model=self.model,
            latents=latents,
            timestep=timesteps,
            caption_features=encoder_hidden_states,
            caption_mask=caption_mask,
        )
        total_loss = l2_highres_loss
        self.log("train/highres_loss", l2_highres_loss, on_step=True, on_epoch=True)

        if self.model_config.use_lowres_loss:
            l2_lowres_loss = self.forward_and_loss(
                model=self.model,
                latents=self.downsample_4x(latents),
                timestep=timesteps,
                caption_features=encoder_hidden_states,
                caption_mask=caption_mask,
            )
            total_loss += l2_lowres_loss
            self.log("train/lowres_loss", l2_lowres_loss, on_step=True, on_epoch=True)

        if self.model_config.use_downsampled_velocity_loss:
            small_velocity = self.downsample_4x(highres_velocity)
            small_target = self.downsample_4x(highres_target)
            l2_velocity_loss = F.mse_loss(
                small_velocity,
                small_target,
                reduction="mean",
            )
            total_loss += l2_velocity_loss
            self.log(
                "train/downsampled_velocity_loss",
                l2_velocity_loss,
                on_step=True,
                on_epoch=True,
            )

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image]:
        prompt: str = batch["prompt"]
        negative_prompt: str | None = batch["negative_prompt"]
        height: int = batch["height"]
        width: int = batch["width"]
        cfg_scale: float = batch["cfg_scale"]
        num_steps: int = batch["num_steps"]
        seed: int = batch["seed"]

        if negative_prompt is None and cfg_scale > 0:
            negative_prompt = ""

        with self.accelerator.autocast():
            image = self.raw_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                cfg_scale=cfg_scale,
                num_inference_steps=num_steps,
                seed=seed,
                max_token_length=self.model_config.max_token_length,
            )[0]

        self.log(
            f"preview/image_{preview_index}",
            wandb_image(image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )

        return [image]

    def after_setup_model(self):
        if self.config.trainer.gradient_checkpointing:
            self.model.denoiser.set_gradient_checkpointing(True)

        super().after_setup_model()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._is_peft:
            return self.model.state_dict()

        state_dict = get_adapter_parameters(self.model)
        state_dict = {convert_to_comfy_key(k): v for k, v in state_dict.items()}
        return state_dict

    def before_setup_model(self):
        pass

    def before_eval_step(self):
        pass

    def before_backward(self):
        pass


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_preview_dataset_class(TextToImagePreviewConfig)
    trainer.register_model_class(Lumina2ForTextToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
