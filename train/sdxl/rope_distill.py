from PIL.Image import Image
import click

import torch
import torch.nn as nn
import torch.nn.functional as F

from accelerate import init_empty_weights
from safetensors.torch import load_file

from src.models.sdxl.adapter.rope import (
    SDXLWithRoPEConfig,
    SDXLWithRoPEModel,
    while_rope_disabled,
    while_rope_enabled,
)
from src.models.sdxl import convert_to_comfy_key, convert_from_original_key
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.diffusion import (
    prepare_noised_latents,
    loss_with_predicted_noise,
)
from src.modules.timestep.sampling import uniform_randint
from src.modules.peft import (
    get_adapter_parameters,
    while_peft_disabled,
    while_peft_enabled,
    load_peft_weight,
)
from src.utils.logging import wandb_image


class SDXLForRoPEDistillTrainingConfig(SDXLWithRoPEConfig):
    max_token_length: int = 225  # 75 * 3

    l2_loss_weight: float = 1.0
    distill_loss_weight: float = 1.0


class SDXLForTextToImageTraining(ModelForTraining, nn.Module):
    model: SDXLWithRoPEModel

    model_config: SDXLForRoPEDistillTrainingConfig
    model_config_class = SDXLForRoPEDistillTrainingConfig

    def setup_model(self):
        with init_empty_weights():
            self.model_config.denoiser.rope_enabled = True  # force RoPE enabled
            self.model = SDXLWithRoPEModel(self.model_config)

            # freeze other modules
            self.model.text_encoder.eval()
            self.model.vae.eval()  # type: ignore

        self.model._from_checkpoint()  # load!

    def load_peft_weights(self):
        if peft_config := self.config.peft:
            if not isinstance(peft_config, list):
                peft_config = [peft_config]
            for peft_target_config in peft_config:
                if (weight_path := peft_target_config.resume_weight_path) is not None:
                    state_dict = load_file(weight_path)
                    load_peft_weight(
                        self.model,
                        {
                            convert_from_original_key(k): v
                            for k, v in state_dict.items()
                        },
                    )
                    self.print(f"Loaded PEFT weights from {weight_path}")

    @property
    def raw_model(self) -> SDXLWithRoPEModel:
        return self.accelerator.unwrap_model(self.model)

    def sanity_check(self):
        latent = self.model.prepare_latents(
            batch_size=1,
            height=96,
            width=96,
            dtype=torch.bfloat16,
            max_noise_sigma=self.model.scheduler.get_max_noise_sigma(
                sigmas=torch.tensor(5.0)
            ),
            device=self.accelerator.device,
        )
        encoder_hidden_states = torch.randn(
            1,
            77,  # max token len
            self.model_config.denoiser.context_dim,  # 2048
            device=self.accelerator.device,
        )
        pooled_hidden_states = torch.randn(
            1,
            1280,  # text encoder 2
            device=self.accelerator.device,
        )
        timestep = torch.tensor([50], dtype=torch.long, device=self.accelerator.device)
        original_size = torch.tensor(
            [96, 96], device=self.accelerator.device
        ).unsqueeze(0)
        target_size = torch.tensor([96, 96], device=self.accelerator.device).unsqueeze(
            0
        )
        crop_coords_top_left = torch.tensor(
            [0, 0], device=self.accelerator.device
        ).unsqueeze(0)

        with self.accelerator.autocast(), while_rope_disabled(self.model):
            _noise_pred = self.model.denoiser(
                latents=latent,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

        with self.accelerator.autocast(), while_rope_enabled(self.model):
            _noise_pred = self.model.denoiser(
                latents=latent,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]
        original_size = batch["original_size"]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_output = self.model.text_encoder.encode_prompts(
                caption,
                max_token_length=self.model_config.max_token_length,
            )
            encoder_hidden_states, pooled_hidden_states = (
                self.model.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=False,
                    device=self.accelerator.device,
                )
            )

            latents = self.model.encode_image(pixel_values)
            timesteps = uniform_randint(
                latents_shape=latents.shape,
                device=self.accelerator.device,
                min_timesteps=0,
                max_timesteps=1000,  # change this for addift?
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3.1 Teacher prediction
        with (
            torch.inference_mode(),
            while_peft_disabled(self.model),
            while_rope_disabled(self.model),
        ):
            teacher_pred = self.model(
                latents=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

        # 3.2 Student prediction
        with while_rope_enabled(self.model):
            student_pred = self.model(
                latents=noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

        # 4. Calculate the loss
        total_loss = torch.tensor(0.0, device=self.accelerator.device)
        if self.model_config.l2_loss_weight > 0:
            l2_loss = loss_with_predicted_noise(
                latents=latents,
                random_noise=random_noise,
                predicted_noise=student_pred,
            )
            l2_loss = l2_loss * self.model_config.l2_loss_weight
            total_loss = total_loss + l2_loss
            self.log("train/l2_loss", l2_loss, on_step=True, on_epoch=True)
        if self.model_config.distill_loss_weight > 0:
            distill_loss = F.mse_loss(
                input=student_pred,
                target=teacher_pred.detach(),
                reduction="mean",
            )
            distill_loss = distill_loss * self.model_config.distill_loss_weight
            total_loss = total_loss + distill_loss
            self.log("train/distill_loss", distill_loss, on_step=True, on_epoch=True)

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
    trainer.register_model_class(SDXLForTextToImageTraining)

    trainer.train()


if __name__ == "__main__":
    main()
