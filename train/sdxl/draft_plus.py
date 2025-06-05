from PIL.Image import Image
import click
from contextlib import nullcontext
import itertools

import torch
import torch.nn as nn

from accelerate import init_empty_weights

from src.models.sdxl import SDXLConfig, SDXLModel, convert_to_comfy_key
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
)
from src.utils.dtype import str_to_dtype
from src.utils.logging import wandb_image
from src.modules.reward import load_reward_models, RewardModelConfig


class SDXLForDRaFTPlusTrainingConfig(SDXLConfig):
    max_token_length: int = 225  # 75 * 3

    truncation_steps: int = 1
    total_steps: int = 25

    reward_models: list[RewardModelConfig]


# ref: https://github.com/NVIDIA/NeMo-Aligner/blob/main/nemo_aligner/models/mm/stable_diffusion/megatron_sdxl_draftp_model.py
class SDXLForDRaFTPlusTraining(ModelForTraining, nn.Module):
    model: SDXLModel

    model_config: SDXLForDRaFTPlusTrainingConfig
    model_config_class = SDXLForDRaFTPlusTrainingConfig

    def setup_model(self):
        with init_empty_weights():
            self.model = SDXLModel(self.model_config)

            # freeze other modules
            self.model.text_encoder.eval()
            self.model.vae.eval()  # type: ignore

        self.model._from_checkpoint()  # load!

        # load reward models
        self.reward_models = load_reward_models(
            self.model_config.reward_models,
            device=self.accelerator.device,
        )

    @property
    def raw_model(self) -> SDXLModel:
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

    def train_step(self, batch: dict) -> torch.Tensor:
        # pixel_values = batch["image"]
        caption = batch["caption"]
        negative_prompt = batch.get("negative_prompt", None)
        cfg_scale = batch["cfg_scale"]

        original_size = batch["original_size"]
        height, width = original_size[0]
        target_size = batch["target_size"]
        crop_coords_top_left = batch["crop_coords_top_left"]
        batch_size = len(caption)
        dtype = str_to_dtype(self.model_config.dtype)

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_output = self.model.text_encoder.encode_prompts(
                caption,
                negative_prompts=negative_prompt,
                use_negative_prompts=True,
                max_token_length=self.model_config.max_token_length,
            )
            encoder_hidden_states, pooled_hidden_states = (
                self.model.prepare_encoder_hidden_states(
                    encoder_output=encoder_output,
                    do_cfg=True,
                    device=self.accelerator.device,
                )
            )
            timesteps, sigmas = self.prepare_timesteps(
                num_inference_steps=self.model_config.total_steps,
                device=self.accelerator.device,
            )

            latents = self.model.prepare_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                dtype=dtype,
                device=self.accelerator.device,
                max_noise_sigma=self.model.scheduler.get_max_noise_sigma(sigmas),
            )
            original_size = original_size.expand(encoder_hidden_states.size(0), -1)
            target_size = target_size.expand(encoder_hidden_states.size(0), -1)
            crop_coords = crop_coords_top_left.expand(encoder_hidden_states.size(0), -1)

        # 2. Denoise (total_steps - truncation_steps) without gradient
        max_no_grad_steps = (
            self.model_config.total_steps - self.model_config.truncation_steps
        )
        draftp_pred_list: list[torch.Tensor] = []  # predictions of DRaFT+ model
        reference_pred_list: list[torch.Tensor] = []  # predictions of the base model
        for i, current_timestep in enumerate(timesteps):
            current_sigma, next_sigma = sigmas[i], sigmas[i + 1]

            # 3. calculate the gradient only during the last truncation_steps
            with torch.no_grad() if i < max_no_grad_steps else nullcontext():
                # expand latents if doing cfg
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.model.scheduler.scale_model_input(
                    latent_model_input, current_sigma
                )

                batch_timestep = current_timestep.expand(latent_model_input.size(0)).to(
                    self.accelerator.device
                )

                # predict noise model_output
                noise_pred = self.model.denoiser(
                    latents=latent_model_input,
                    timestep=batch_timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_pooler_output=pooled_hidden_states,
                    original_size=original_size,
                    target_size=target_size,
                    crop_coords_top_left=crop_coords,
                )

                # do CFG
                noise_pred_positive, noise_pred_negative = noise_pred.chunk(2)
                noise_pred = noise_pred_negative + cfg_scale * (
                    noise_pred_positive - noise_pred_negative
                )
                latents = self.model.scheduler.ancestral_step(
                    latents,
                    noise_pred,
                    current_sigma,
                    next_sigma,
                )

                if i < max_no_grad_steps:
                    continue

                # 3.5. last truncation_steps
                # save the predictions
                draftp_pred_list.append(noise_pred)

                # calculate the reference prediction
                with torch.no_grad(), while_peft_disabled(self.model):
                    reference_noise_pred = self.model.denoiser(
                        latents=latent_model_input,
                        timestep=batch_timestep,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_pooler_output=pooled_hidden_states,
                        original_size=original_size,
                        target_size=target_size,
                        crop_coords_top_left=crop_coords,
                    )
                reference_pred_list.append(reference_noise_pred)

        # 4. stack the predictions
        draftp_pred = torch.stack(draftp_pred_list, dim=1)
        reference_pred = torch.stack(reference_pred_list, dim=1)

        # 5. decode the clean latents
        with torch.no_grad():
            images = self.model.decode_image(latents)

        # 7. call reward functions
        reward_scores = torch.stack(
            [reward_model(images, caption) for reward_model in self.reward_models],
            dim=0,
        )  # [num_rewards, batch_size]

        # TODO!: どうにかして報酬から勾配計算するよ

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
    trainer.register_model_class(SDXLForDRaFTPlusTraining)

    trainer.train()


if __name__ == "__main__":
    main()
