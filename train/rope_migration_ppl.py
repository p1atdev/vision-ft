import click
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
from src.dataset.single_caption_bucket import SingleCaptionDatasetConfig
from src.modules.peft import (
    get_adapter_parameters,
    while_peft_disabled,
)
from src.modules.positional_encoding.rope import RoPEFrequency
from src.modules.timestep import sigmoid_randn
from src.modules.migration.scale import MigrationScaleFromZero


class DenoiserForRoPEMigration(Denoiser):
    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__(config)

        self.migration_scale = MigrationScaleFromZero(dim=1)

    def forward(
        self,
        latent: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

            # get migration scale
            base_freqs = torch.ones_like(rope_freqs)
            base_freqs[..., 1] = 0  # base_freqs does not rotate
            difference = base_freqs - rope_freqs
            rope_freqs = base_freqs - self.migration_scale.scale_positive(difference)

            # add scaled position encoding
            patches = patches + self.migration_scale.scale_negative(
                self.get_pos_encoding(height, width)
            )
        else:
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
        return noise_prediction, self.migration_scale.scale

    @contextmanager
    def while_rope_disabled(self):
        self.use_rope = False
        yield
        self.use_rope = True


class AuraFlorForRoPEMigration(AuraFlowModel):
    denoiser: DenoiserForRoPEMigration
    denoiser_class = DenoiserForRoPEMigration


class AuraFlowForRoPEMigrationTraining(ModelForTraining, nn.Module):
    model: AuraFlorForRoPEMigration

    model_config: AuraFlowConig
    model_config_class = AuraFlowConig

    def sanity_check(self):
        # migration scale must be trainable
        assert any(
            "migration_scale.scale" in n and p.requires_grad is True
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
            assert (
                self.model_config.denoiser.use_rope
            ), "This model is not for positional attention training"
            with init_empty_weights():
                self.model = AuraFlorForRoPEMigration(self.model_config)

            self.model._load_original_weights()
            assert self.model.denoiser.migration_scale.scale.requires_grad is True

    def train_step(self, batch: dict) -> torch.Tensor:
        caption = batch["caption"]
        height = batch["height"][0]
        width = batch["width"][0]

        # 1. Prepare the inputs
        with torch.no_grad():
            encoder_hidden_states = self.model.text_encoder.encode_prompts(
                caption
            ).positive_embeddings
            noise_latents = self.model.prepare_latents(
                batch_size=encoder_hidden_states.size(0),
                height=height,
                width=width,
                dtype=encoder_hidden_states.dtype,
                device=self.accelerator.device,
            )
            timesteps = sigmoid_randn(
                latents_shape=noise_latents.shape,
                device=self.accelerator.device,
            )

        # 3. Predict the noise
        # prior prediction
        with while_peft_disabled(self.model):
            with self.model.denoiser.while_rope_disabled():
                with torch.inference_mode():
                    prior_preds, _ = self.model.denoiser(
                        latent=noise_latents,
                        encoder_hidden_states=encoder_hidden_states,
                        timestep=timesteps,
                    )

        # prediction with peft enabled
        actual_preds, migration_scale = self.model.denoiser(
            latent=noise_latents,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timesteps,
        )

        # 4. Calculate loss
        # Prior preservation loss (ppl)
        ppl_loss = F.mse_loss(
            prior_preds.detach(),
            actual_preds,  # calculate the loss only this one
            reduction="mean",
        )
        # want the scale to be 1
        rope_migration_loss = F.mse_loss(
            migration_scale,
            torch.ones_like(migration_scale),
            reduction="mean",
        )
        total_loss = ppl_loss + rope_migration_loss

        self.log("train/loss", total_loss, on_step=True, on_epoch=True)
        self.log(
            "train/ppl_loss",
            ppl_loss,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train/rope_migration_loss",
            rope_migration_loss,
            on_step=True,
            on_epoch=True,
        )

        return total_loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def before_setup_model(self):
        super().before_setup_model()

    def after_setup_model(self):
        if self.accelerator.is_main_process:
            if self.config.trainer.gradient_checkpointing:
                self.model.denoiser._set_gradient_checkpointing(True)

        # make migration scale trainable
        for n, p in self.model.named_parameters():
            if "migration_scale.scale" in n:
                p.requires_grad = True

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
    trainer.register_train_dataset_class(SingleCaptionDatasetConfig)
    trainer.register_model_class(AuraFlowForRoPEMigrationTraining)

    trainer.train()


if __name__ == "__main__":
    main()
