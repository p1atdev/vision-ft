from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.utils.data as data

from accelerate import Accelerator
from transformers import set_seed

from ..config import TrainConfig, DEBUG_MODE_TYPE
from ..saving import ModelSavingStrategy, get_saving_callback
from ..preview import PreviewStrategy, get_preview_callback
from ..dataset.util import DatasetConfig
from ..dataloader import get_dataloader_for_bucketing, get_dataloader_for_preview
from ..utils.logging import get_trackers
from ..utils.safetensors import load_file_with_rename_key_map
from ..models.for_training import ModelForTraining
from ..modules.peft import (
    PeftConfigMixin,
    replace_to_peft_layer,
    print_trainable_parameters,
    load_peft_weight,
)


class Trainer:
    model: ModelForTraining

    peft_config: PeftConfigMixin | None

    dataset_config: DatasetConfig
    preview_dataset_config: DatasetConfig | None

    train_dataloader: data.DataLoader
    eval_dataloader: data.DataLoader | None = None
    preview_dataloader: data.DataLoader | None = None

    debug_mode: DEBUG_MODE_TYPE

    def __init__(
        self,
        config: TrainConfig,
        seed: int = 42,
    ) -> None:
        self.config = config
        self.peft_config = config.peft

        self.seed = seed
        self.debug_mode = config.trainer.debug_mode

        self.accelerator = Accelerator(
            log_with=get_trackers(config),
            gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        )
        if self.debug_mode is False and (tracker := config.tracker) is not None:
            self.accelerator.init_trackers(
                project_name=tracker.project_name,
                config=config.model_dump(),
            )

    def register_model_class(self, model_cls, *args, **kwargs):
        self.model_cls = model_cls
        self.model = model_cls(self.accelerator, self.config, *args, **kwargs)

    def register_train_dataset_class(
        self, dataset_config_class: type[DatasetConfig], *args, **kwargs
    ):
        self.dataset_config = dataset_config_class.model_validate(self.config.dataset)

    def register_preview_dataset_class(
        self, dataset_config_class: type[DatasetConfig], *args, **kwargs
    ):
        if self.config.preview is not None:
            self.preview_dataset_config = dataset_config_class.model_validate(
                self.config.preview.data
            )

    def get_saving_callbacks(self):
        if (saving := self.config.saving) is not None:
            if len(saving.callbacks) == 0:
                warnings.warn("No saving callbacks found in the config")
            return [get_saving_callback(callback) for callback in saving.callbacks]

        self.accelerator.print("No saving config. Model will not be saved.")
        return []

    def get_preview_callbacks(self):
        if (preview := self.config.preview) is not None:
            if len(preview.callbacks) == 0:
                warnings.warn("No preview callbacks found in the config")
            return [get_preview_callback(callback) for callback in preview.callbacks]

        self.accelerator.print("No preview config. Model will not be saved.")
        return []

    def prepare_dataloaders(self):
        train_ds = self.dataset_config.get_dataset()

        train_dataloader = get_dataloader_for_bucketing(
            train_ds,
            shuffle=self.dataset_config.shuffle,
            num_workers=self.dataset_config.num_workers,
        )
        self.train_dataloader = self.accelerator.prepare_data_loader(train_dataloader)
        # TODO: eval
        # eval_dataloader = get_dataloader(
        #     eval_ds,
        #     batch_size=self.dataset_config.batch_size,
        #     shuffle=False,
        #     num_workers=self.dataset_config.num_workers,
        # )
        if (preview_config := self.config.preview) is not None:
            self.print("Preview config found. Preparing preview dataloader...")
            preview_dataloader = get_dataloader_for_preview(
                preview_config.data.get_dataset(),
                # TODO: other args
            )
            self.preview_dataloader = self.accelerator.prepare_data_loader(
                preview_dataloader
            )

    def prepare_saving_strategy(self):
        if (saving := self.config.saving) is not None:
            self.saving_strategy = ModelSavingStrategy.from_config(
                config=saving.strategy,
                steps_per_epoch=len(self.train_dataloader),
                total_epochs=self.config.num_train_epochs,
            )
        else:
            self.saving_strategy = ModelSavingStrategy(
                steps_per_epoch=len(self.train_dataloader),
                total_epochs=self.config.num_train_epochs,
                per_epochs=None,
                per_steps=None,
                save_last=False,
            )
        self.saving_callbacks = self.get_saving_callbacks()

    def prepare_preview_strategy(self):
        if (preview := self.config.preview) is not None:
            self.preview_strategy = PreviewStrategy.from_config(
                config=preview.strategy,
                steps_per_epoch=len(self.train_dataloader),
                total_epochs=self.config.num_train_epochs,
            )
        else:
            self.preview_strategy = PreviewStrategy(
                steps_per_epoch=len(self.train_dataloader),
                total_epochs=self.config.num_train_epochs,
                per_epochs=None,
                per_steps=None,
            )
        self.preview_callbacks = self.get_preview_callbacks()

    def setup_peft_if_needed(self):
        if self.peft_config is not None:
            self.print("Applying PEFT")
            self.model._set_is_peft(True)
            replace_to_peft_layer(
                self.model,
                self.peft_config,
            )
            if (weight_path := self.peft_config.resume_weight_path) is not None:
                load_peft_weight(
                    self.model,
                    load_file_with_rename_key_map(
                        weight_path, self.peft_config.resume_rename_key_map
                    ),
                )
            print_trainable_parameters(self.model, self.print)
        else:
            self.model._set_is_peft(False)

    def prepare_model(self):
        if self.accelerator.is_main_process:
            self.model.before_setup_model()
            self.model.setup_model()
            self.setup_peft_if_needed()
            self.model.after_setup_model()

        self.model = self.accelerator.prepare(self.model)

    def before_train(self):
        self.torch_configuration()

        if self.debug_mode is not False:
            self.print(f"Debug mode is enabled: {self.debug_mode}")

        self.print("before_train()")
        self.print(f"Seed: {self.seed}")
        set_seed(self.seed)

        self.print("Setting up dataloaders")
        self.prepare_dataloaders()

        self.print("Setting up saving strategy")
        self.prepare_saving_strategy()

        self.print("Setting up preview strategy")
        self.prepare_preview_strategy()

        if self.debug_mode == "dataset":
            self.debug_dataset()
            self.print("Dataset check done. Exiting...")
            return

        self.print("Setting up model")
        self.prepare_model()
        self.print("Setting up optimizer")
        self.model.setup_optimizer()

    def after_train(self):
        self.print("after_train()")
        pass

    def training_loop(self):
        self.print("training_loop()")

        current_step = 0
        total_epochs = self.config.num_train_epochs

        for epoch in range(1, total_epochs + 1):  # shift to 1-indexed
            self.model.before_train_epoch()

            with tqdm(
                total=len(self.train_dataloader), desc=f"Train Epoch {epoch}"
            ) as pbar:
                for _steps, batch in enumerate(self.train_dataloader):
                    with self.accelerator.accumulate(self.model):
                        current_step += 1
                        self.model.before_train_step()

                        with self.accelerator.autocast():
                            loss = self.model.train_step(batch)
                        self.model.backward(loss)
                        pbar.set_postfix({"loss": loss.item()})

                        pbar.update(1)

                        self.call_saving_callbacks(epoch, current_step)
                        self.call_preview_callbacks(epoch, current_step)

                        self.model.after_train_step()

                        if self.debug_mode == "1step":
                            break

            self.model.after_train_epoch()
            self.model.log("epoch", epoch)

            if self.eval_dataloader is not None:
                self.model.before_eval_epoch()

                with tqdm(
                    total=len(self.eval_dataloader), desc=f"Eval Epoch {epoch}"
                ) as pbar:
                    for _steps, batch in enumerate(self.eval_dataloader):
                        self.model.before_eval_step()

                        with self.accelerator.autocast():
                            loss = self.model.eval_step(batch)
                        pbar.set_postfix({"loss": loss.item()})

                        pbar.update(1)

                        self.model.after_eval_step()

                        if self.debug_mode == "1step":
                            break

                self.model.after_eval_epoch()

            if self.debug_mode == "1step":
                break

    def call_saving_callbacks(self, epoch: int, steps: int):
        if self.saving_strategy.should_save(epoch, steps):
            self.model.before_save_model()

            if len(self.saving_callbacks) > 0:
                if self.accelerator.is_main_process:
                    unwrapped_model: ModelForTraining = self.accelerator.unwrap_model(
                        self.model
                    )
                    state_dict = unwrapped_model.get_state_dict_to_save()
                    self.print("Saving model...")

                    for callback in self.saving_callbacks:
                        callback.save_state_dict(state_dict, epoch, steps)

                    self.print("Model saved.")

            self.accelerator.wait_for_everyone()
            self.model.after_save_model()

    def call_preview_callbacks(self, epoch: int, steps: int):
        if self.preview_strategy.should_preview(epoch, steps):
            self.model.before_preview()

            if len(self.preview_callbacks) > 0:
                assert self.preview_dataloader is not None
                self.print("Generating preview images...")
                for i, batch in tqdm(
                    enumerate(self.preview_dataloader),
                    total=len(self.preview_dataloader),
                    desc="Preview",
                ):
                    self.model.before_preview_step()
                    preview = self.model.preview_step(batch, preview_index=i)
                    for callback in self.preview_callbacks:
                        callback.preview_image(preview, epoch, steps, i, metadata=batch)
                    self.model.after_preview_step()

                self.print("Preview done.")

            self.accelerator.wait_for_everyone()
            self.model.after_preview()

    def debug_dataset(self):
        if self.train_dataloader is None:
            raise ValueError("train_dataloader is not prepared")

        self.print("debugging train_dataloader...")
        for batch in self.train_dataloader:
            self.print(batch)

        if self.eval_dataloader is not None:
            self.print("debugging eval_dataloader...")
            for batch in self.eval_dataloader:
                self.print(batch)

    def torch_configuration(self):
        if self.config.trainer.fp32_matmul_precision is not None:
            torch.set_float32_matmul_precision(
                self.config.trainer.fp32_matmul_precision
            )

        if self.config.trainer.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    # User-facing method
    def train(self):
        self.before_train()
        if self.debug_mode == "dataset":
            return

        self.model.sanity_check()
        if self.debug_mode == "sanity_check":
            self.print("Sanity check done. Exiting...")
            return

        self.training_loop()

        self.after_train()

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.debug_mode is False:
            return

        self.accelerator.log(*args, **kwargs)
