from PIL import Image
from typing import Literal
import click

import torch
import torch.nn as nn
from accelerate import init_empty_weights

from src.models.sdxl.adapter.prompt_free import (
    SDXLModelWithPFG,
    SDXLModelWithPFGConfig,
)
from src.models.for_training import ModelForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.dataset.preview.text_to_image import TextToImagePreviewConfig
from src.modules.loss.diffusion import (
    prepare_noised_latents,
    loss_with_predicted_noise,
)
from src.modules.timestep.sampling import uniform_randint, gaussian_randint
from src.modules.peft import get_adapter_parameters

from src.utils.tensor import remove_orig_mod_prefix
from src.utils.logging import wandb_image


class SDXLModelWithPFGTrainingConfig(SDXLModelWithPFGConfig):
    max_token_length: int = 75  # 75, 150, 225, ... (75 * N)
    drop_image_rate: float = 0.1

    freeze_vision_encoder: bool = True

    timestep_sampling: Literal["uniform", "gaussian"] = "uniform"
    timestep_sampling_args: dict = {}


class SDXLIPAdapterTraining(ModelForTraining, nn.Module):
    model: SDXLModelWithPFG

    model_config: SDXLModelWithPFGTrainingConfig
    model_config_class = SDXLModelWithPFGTrainingConfig

    def setup_model(self):
        # setup SDXL
        with init_empty_weights():
            self.model = SDXLModelWithPFG(self.model_config)

            self.model.text_encoder.eval()
            self.model.text_encoder.requires_grad_(False)
            self.model.vae.eval()  # type: ignore
            self.model.vae.requires_grad_(False)

        self.model._from_checkpoint()  # load!

        if self.config.peft is None:
            self.model.denoiser.eval()
            self.model.denoiser.requires_grad_(False)

        # make adapter trainable later!
        # self.model.manager.set_adapter_trainable(True)

        if self.model_config.freeze_vision_encoder:
            # freeze vision encoder
            self.model.vision_encoder.eval()
            self.model.vision_encoder.requires_grad_(False)
        else:
            # make vision encoder trainable
            self.model.vision_encoder.train()
            self.model.vision_encoder.requires_grad_(True)

    @property
    def raw_model(self) -> SDXLModelWithPFG:
        return self.accelerator.unwrap_model(self.model)

    @torch.no_grad()
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
            (
                self.model_config.max_token_length
                + self.model_config.adapter.num_image_tokens
            ),  # max token len + PFG token
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

        # print(self.raw_model)

        with self.accelerator.autocast():
            _noise_pred = self.model.denoiser(
                latents=latent,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_pooler_output=pooled_hidden_states,
                original_size=original_size,
                target_size=target_size,
                crop_coords_top_left=crop_coords_top_left,
            )

    def sample_timestep(self, shape: torch.Size) -> torch.IntTensor:
        args = self.model_config.timestep_sampling_args
        if self.model_config.timestep_sampling == "uniform":
            return uniform_randint(
                latents_shape=shape,
                device=self.accelerator.device,
                min_timesteps=args.get("min_timesteps", 0),
                max_timesteps=args.get("max_timesteps", 1000),
            )
        elif self.model_config.timestep_sampling == "gaussian":
            return gaussian_randint(
                latents_shape=shape,
                device=self.accelerator.device,
                min_timesteps=args.get("min_timesteps", 0),
                max_timesteps=args.get("max_timesteps", 1000),
                mean=args.get("mean", 100),
                std=args.get("std", 100),
            )

        raise ValueError(
            f"Invalid sampling type: {self.model_config.timestep_sampling}"
        )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]
        original_size = batch["original_size"]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]

        # 1. Prepare the inputs
        with torch.no_grad():
            reference_pixel_values = self.model.preprocess_reference_image(pixel_values)
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
            timesteps = self.sample_timestep(latents.shape)
            drop_image = (
                torch.rand(
                    reference_pixel_values.size(0), device=self.accelerator.device
                )
                < self.model_config.drop_image_rate
            )

        # pfg reference inputs
        image_tokens: torch.Tensor = self.model.encode_reference_image(
            reference_pixel_values
        )
        # drop ip tokens randomly for cfg
        image_tokens[drop_image] = 0

        # cat with seq len to pass through the model
        encoder_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                image_tokens,
            ],
            dim=1,  # seq len
        )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3. Predict the noise
        noise_pred = self.model(
            latents=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_pooler_output=pooled_hidden_states,
            original_size=original_size,
            target_size=target_size,
            crop_coords_top_left=crop_coords_top_left,
        )

        # 4. Calculate the loss
        l2_loss = loss_with_predicted_noise(
            latents=latents,
            random_noise=random_noise,
            predicted_noise=noise_pred,
        )
        total_loss = l2_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @torch.inference_mode()
    def preview_step(self, batch, preview_index: int) -> list[Image.Image]:
        prompt: str = batch["prompt"]
        negative_prompt: str | None = batch["negative_prompt"]
        height: int = batch["height"]
        width: int = batch["width"]
        cfg_scale: float = batch["cfg_scale"]
        num_steps: int = batch["num_steps"]
        seed: int = batch["seed"]
        extra: dict = batch["extra"]
        reference_image_path: str = extra["reference_image_path"]

        reference_pil = Image.open(reference_image_path).convert("RGB")
        reference_image = self.model.preprocess_reference_image(reference_pil)

        if negative_prompt is None and cfg_scale > 0:
            negative_prompt = ""

        with self.accelerator.autocast():
            image = self.raw_model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                reference_image=reference_image,
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
        # set projector trainable
        self.model.manager.set_adapter_trainable(True)

        if self.config.trainer.gradient_checkpointing:
            self.model.denoiser.set_gradient_checkpointing(True)

        super().after_setup_model()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        # get pfg adapter state dict
        adapter_state_dict = self.raw_model.manager.get_state_dict()

        state_dict = adapter_state_dict

        # LoRA
        if self._is_peft:
            peft_state_dict = get_adapter_parameters(self.model)
            state_dict |= peft_state_dict

        state_dict = {remove_orig_mod_prefix(k): v for k, v in state_dict.items()}

        return state_dict

    def get_metadata_to_save(self) -> dict[str, str]:
        if self.model_config.adapter.projector_type == "resampler":
            return {
                "num_heads": str(
                    self.model_config.adapter.projector_args.get(
                        "num_heads",
                        8,
                    )
                ),
            }
        return {}

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
    trainer.register_model_class(SDXLIPAdapterTraining)

    trainer.train()


if __name__ == "__main__":
    main()
