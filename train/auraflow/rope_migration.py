import click
from PIL import Image
from typing import Literal
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from accelerate import init_empty_weights

from src.models.auraflow.config import DenoiserConfig
from src.models.auraflow import AuraFlowModel, AuraFlowConig, convert_to_comfy_key
from src.models.for_training import ModelForTraining
from src.models.auraflow.denoiser import Denoiser, modulate
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig
from src.modules.peft import get_adapter_parameters, while_peft_disabled
from src.modules.positional_encoding.rope import RoPEFrequency
from src.modules.timestep import sigmoid_randn, sample_timestep
from src.modules.loss.flow_match import (
    prepare_noised_latents,
    loss_with_predicted_velocity,
)
from src.modules.migration.scale import MigrationScaleFromZero
from src.utils.logging import wandb_image


class DenoiserForRoPEMigration(Denoiser):
    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__(config)

        self.migration_scale = MigrationScaleFromZero(dim=1)
        self.migration = True

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _in_channels, height, width = latent.shape

        # 1. condition sequence
        cond_sequences = encoder_hidden_states[:batch_size]
        cond_tokens = self.cond_seq_linear(cond_sequences)
        cond_tokens = torch.cat(
            [self.register_tokens.repeat(cond_tokens.size(0), 1, 1), cond_tokens], dim=1
        )
        timestep = timestep[:batch_size]
        global_cond = self.t_embedder(timestep)

        # 2. patchify
        patches = self.patchify(latent)
        patches = self.init_x_linear(patches)  # project

        # 2.5 prepare RoPE migration
        assert isinstance(self.rope_frequency, RoPEFrequency)
        if self.use_rope:
            # RoPE
            text_token_indices = self.rope_frequency.get_text_position_indices(
                cond_tokens.size(1)
            )
            image_token_indices = self.rope_frequency.get_image_position_indices(
                height, width
            )
            token_indices = torch.cat(
                [text_token_indices, image_token_indices], dim=0
            ).to(self.device)
            rope_freqs = self.rope_frequency(token_indices)

            if self.migration:
                # get migration scale
                base_freqs = torch.ones_like(rope_freqs, device=rope_freqs.device)
                base_freqs[..., 1] = 0  # base_freqs does not rotate
                difference = base_freqs - rope_freqs
                rope_freqs = base_freqs - self.migration_scale.scale_positive(
                    difference,
                )

                # add scaled position encoding
                patches = patches + self.migration_scale.scale_negative(
                    self.get_pos_encoding(height, width),
                )
        else:
            # learned position encoding
            rope_freqs = None
            patches = patches + self.get_pos_encoding(height, width)

        # for gradient checkpointing
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)

            return custom_forward

        # 3. double layers
        if len(self.double_layers) > 0:
            for layer in self.double_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    cond_tokens, patches = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        cond_tokens,
                        patches,
                        global_cond,
                        rope_freqs,
                        use_reentrant=False,
                    )
                else:
                    cond_tokens, patches = layer(
                        cond_tokens, patches, global_cond, rope_freqs
                    )

        # 4. single layers
        if len(self.single_layers) > 0:
            cond_tokens_len = cond_tokens.size(1)
            context = torch.cat([cond_tokens, patches], dim=1)
            for layer in self.single_layers:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    context = checkpoint.checkpoint(  # type: ignore
                        create_custom_forward(layer),
                        context,
                        global_cond,
                        rope_freqs,
                        use_reentrant=False,
                    )
                else:
                    context = layer(context, global_cond, rope_freqs)
            assert isinstance(context, torch.Tensor)

            # take only patches
            patches = context[:, cond_tokens_len:]

        # 5. modulate
        f_shift, f_scale = self.modF(global_cond).chunk(2, dim=1)
        patches = modulate(patches, f_shift, f_scale)
        patches = self.final_linear(patches)

        # 6. unpatchify
        noise_prediction = self.unpatchify(
            patches, height // self.patch_size, width // self.patch_size
        )
        return noise_prediction


class AuraFlowForRoPEMigration(AuraFlowModel):
    denoiser: DenoiserForRoPEMigration
    denoiser_class = DenoiserForRoPEMigration

    @contextmanager
    def while_rope_disabled(self):
        tmp = self.denoiser.use_rope
        self.denoiser.use_rope = False
        yield
        self.denoiser.use_rope = tmp

    @contextmanager
    def while_migration_disabled(self):
        tmp = self.denoiser.migration
        self.denoiser.migration = False
        yield
        self.denoiser.migration = tmp  # restore


class AuraFlowForRoPEMigrationConfig(AuraFlowConig):
    # loss flags
    noise_prediction_loss: bool = True  # normal text to image training loss
    migration_loss: bool = True  # gradually migrate to RoPE
    prior_preservation_loss: bool = False  # preserve the prior

    migration_freezing_threshold: float | None = 1e-7
    timestep_sampling: Literal["sigmoid", "uniform"] = "sigmoid"


class AuraFlowForRoPEMigrationTraining(ModelForTraining, nn.Module):
    model: AuraFlowForRoPEMigration

    model_config: AuraFlowForRoPEMigrationConfig
    model_config_class = AuraFlowForRoPEMigrationConfig

    def sanity_check(self):
        # migration scale must be trainable
        assert any(
            "migration_scale.scale" in n
            and p.requires_grad is self.model_config.migration_loss
            for n, p in self.model.named_parameters()
        )

        latent = self.model.prepare_latents(
            batch_size=1,
            height=96,
            width=96,
            dtype=torch.bfloat16,
            device=self.accelerator.device,
        )
        prompt = torch.randn(
            1,
            256,  # max token len
            self.model_config.denoiser.joint_attention_dim,
            device=self.accelerator.device,
        )
        timestep = torch.tensor([0.5], device=self.accelerator.device)

        with self.accelerator.autocast():
            _noise_pred = self.model.denoiser(
                latent=latent,
                encoder_hidden_states=prompt,
                timestep=timestep,
            )

    def setup_model(self):
        if self.accelerator.is_main_process:
            assert self.model_config.denoiser.use_rope, (
                "This model is not for positional attention training"
            )
            with init_empty_weights():
                self.model = AuraFlowForRoPEMigration(self.model_config)

            self.model._load_original_weights()

            # freeze other modules
            self.model.text_encoder.eval()
            self.model.vae.eval()  # type: ignore

            # set threshold
            if self.model_config.migration_loss:
                self.model.denoiser.migration_scale.freezing_threshold = (
                    self.model_config.migration_freezing_threshold
                )
                self.print(
                    "Migration freezing threshold is set to",
                    self.model_config.migration_freezing_threshold,
                )
            else:
                self.print("Migration loss is disabled")
                self.model.denoiser.migration = False
                self.model.denoiser.migration_scale.freezing_threshold = 2.0
                self.model.denoiser.migration_scale.scale.data = torch.ones_like(
                    self.model.denoiser.migration_scale.scale
                )

    def train_step(self, batch: dict) -> torch.Tensor:
        pixel_values = batch["image"]
        caption = batch["caption"]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder.encode_prompts(
                caption
            ).positive_embeddings
            latents = self.model.encode_image(pixel_values)
            timesteps = sample_timestep(
                latents_shape=latents.shape,
                device=self.accelerator.device,
                sampling_type=self.model_config.timestep_sampling,
            )

        # 2. Prepare the noised latents
        noisy_latents, random_noise = prepare_noised_latents(
            latents=latents,
            timestep=timesteps,
        )

        # 3. Predict the noise
        velocity_pred = self.model.denoiser(
            latent=noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )
        rope_scale = self.model.denoiser.migration_scale.inner_scale
        if self.model_config.prior_preservation_loss:
            with while_peft_disabled(self.model):
                with self.model.while_rope_disabled():
                    preserved_noise_pred, _ = self.model.denoiser(
                        latent=noisy_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timesteps,
                    )

        # 4. Calculate the loss
        loss_dict: dict[str, torch.Tensor] = {}

        if self.model_config.noise_prediction_loss:
            l2_loss = loss_with_predicted_velocity(
                latents=latents,
                random_noise=random_noise,
                predicted_velocity=velocity_pred,
            )
            loss_dict["l2_loss"] = l2_loss

        if self.model_config.migration_loss:
            # want the scale to be 1
            rope_migration_loss = F.mse_loss(
                rope_scale,
                torch.ones_like(rope_scale),
                reduction="mean",
            )
            loss_dict["rope_migration_loss"] = rope_migration_loss

        if self.model_config.prior_preservation_loss:
            ppl_loss = F.mse_loss(
                preserved_noise_pred,
                velocity_pred,
                reduction="mean",
            )
            loss_dict["ppl_loss"] = ppl_loss

        # sum except None
        assert len(loss_dict) > 0, "At least one loss should be enabled"
        total_loss = sum(filter(None, loss_dict.values()))
        assert isinstance(total_loss, torch.Tensor)

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)
        for loss_name, loss_value in loss_dict.items():
            self.log(f"train/{loss_name}", loss_value, on_step=True, on_epoch=True)

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

        if negative_prompt is None and cfg_scale > 0:
            negative_prompt = ""

        with self.accelerator.autocast():
            migration_image = self.model.generate(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                cfg_scale=cfg_scale,
                num_inference_steps=num_steps,
                seed=seed,
            )[0]

            with self.model.while_rope_disabled():
                rope_disabled_image = self.model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_steps,
                    seed=seed,
                )[0]

            with self.model.while_migration_disabled():
                rope_enabled_image = self.model.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    cfg_scale=cfg_scale,
                    num_inference_steps=num_steps,
                    seed=seed,
                )[0]

        self.log(
            f"preview/rope_enabled_{preview_index}",
            wandb_image(rope_enabled_image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            f"preview/rope_disabled_{preview_index}",
            wandb_image(rope_disabled_image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )
        self.log(
            f"preview/migration_{preview_index}",
            wandb_image(migration_image, caption=prompt),
            on_step=True,
            on_epoch=False,
        )

        return [rope_enabled_image, rope_disabled_image, migration_image]

    def before_setup_model(self):
        super().before_setup_model()

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                self.model.denoiser._set_gradient_checkpointing(True)

        # make migration scale trainable
        for n, p in self.model.named_parameters():
            if "migration_scale.scale" in n:
                p.requires_grad = self.model_config.migration_loss

        super().after_setup_model()

    def before_eval_step(self):
        super().before_eval_step()

    def before_backward(self):
        super().before_backward()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        if not self._is_peft:
            return self.model.state_dict()

        state_dict = get_adapter_parameters(self.model)
        state_dict = {convert_to_comfy_key(k): v for k, v in state_dict.items()}
        return state_dict


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_train_dataset_class(TextToImageDatasetConfig)
    trainer.register_model_class(AuraFlowForRoPEMigrationTraining)

    trainer.train()


if __name__ == "__main__":
    main()
