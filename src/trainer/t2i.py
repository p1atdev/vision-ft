from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.utils.data as data

from accelerate import Accelerator
from transformers import set_seed

from ..config import TrainConfig
from ..saving import ModelSavingStrategy, get_saving_callback
from ..dataset.util import DatasetConfig
from ..dataset.bucket import bucketing_collate_fn
from ..dataloader import get_dataloader
from ..utils.logging import get_trackers
from ..models.for_training import ModelForTraining
from ..modules.peft import PeftConfigMixin, replace_to_peft_linear


class Trainer:
    model: ModelForTraining

    peft_config: PeftConfigMixin | None

    dataset_config: DatasetConfig
    dataset_config_class: type[DatasetConfig]

    train_dataloader: data.DataLoader
    eval_dataloader: data.DataLoader | None

    def __init__(
        self,
        config: TrainConfig,
        seed: int = 42,
        only_sanity_check: bool = False,
    ) -> None:
        self.config = config
        self.peft_config = config.peft

        self.seed = seed
        self.only_sanity_check = only_sanity_check

        self.accelerator = Accelerator(
            log_with=get_trackers(config.trackers)
            if not only_sanity_check and config.trackers is not None
            else [],
        )

    def register_model_class(self, model_cls, *args, **kwargs):
        self.model_cls = model_cls
        self.model = model_cls(self.accelerator, self.config, *args, **kwargs)

    def register_dataset_class(
        self, dataset_config_class: type[DatasetConfig], *args, **kwargs
    ):
        self.dataset_config_class = dataset_config_class
        self.dataset_config = dataset_config_class.model_validate(self.config.dataset)

    def get_saving_callbacks(self):
        if (saving := self.config.saving) is not None:
            if len(saving.callbacks) == 0:
                warnings.warn("No saving callbacks found in the config")
            return [get_saving_callback(callback) for callback in saving.callbacks]

        self.accelerator.print("No saving config. Model will not be saved.")
        return []

    def prepare_dataloaders(self):
        train_ds = self.dataset_config.get_dataset()

        train_dataloader = get_dataloader(
            train_ds,
            batch_size=1,
            shuffle=self.dataset_config.shuffle,
            num_workers=self.dataset_config.num_workers,
            collate_fn=bucketing_collate_fn,
        )
        # TODO: eval, generation check
        # eval_dataloader = get_dataloader(
        #     eval_ds,
        #     batch_size=self.dataset_config.batch_size,
        #     shuffle=False,
        #     num_workers=self.dataset_config.num_workers,
        # )
        self.train_dataloader = self.accelerator.prepare_data_loader(train_dataloader)

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

    def setup_peft_if_needed(self):
        if self.peft_config is not None:
            replace_to_peft_linear(
                self.model,
                self.peft_config,
            )

    def before_train(self):
        self.print("before_train()")
        self.print(f"Seed: {self.seed}")
        set_seed(self.seed)

        self.print("Setting up dataloaders")
        self.prepare_dataloaders()

        self.print("Setting up saving strategy")
        self.prepare_saving_strategy()

        self.print("Setting up model")
        self.model.before_setup_model()
        self.model.setup_model()
        self.setup_peft_if_needed()
        self.model.after_setup_model()
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

            for steps, batch in tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f"Train Epoch {epoch}",
            ):
                current_step += 1
                self.model.before_train_step()

                loss = self.model.train_step(batch)
                self.model.backward(loss)

                self.model.after_train_step()

                self.call_saving_callbacks(epoch, current_step)

                if self.only_sanity_check:
                    break

            self.model.after_train_epoch()
            self.model.log("epoch", epoch)

            if self.eval_dataloader is not None:
                self.model.before_eval_epoch()

                for batch in tqdm(self.eval_dataloader, desc=f"Eval Epoch {epoch}"):
                    self.model.before_eval_step()

                    loss = self.model.eval_step(batch)

                    self.model.after_eval_step()

                    if self.only_sanity_check:
                        break

                self.model.after_eval_epoch()

            if self.only_sanity_check:
                break

    def call_saving_callbacks(self, epoch: int, steps: int):
        if self.saving_strategy.should_save(epoch, steps):
            self.model.before_save_model()

            for callback in self.saving_callbacks:
                callback.save(self.model.model, epoch, steps)

            self.model.after_save_model()

    def train(self):
        self.before_train()

        self.model.sanity_check()

        self.training_loop()

        self.after_train()

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.only_sanity_check:
            return

        self.accelerator.log(*args, **kwargs)
