from abc import ABC, abstractmethod
from pydantic import BaseModel
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from accelerate import Accelerator
from transformers import set_seed

from .optimizer import get_optimizer
from .scheduler import get_scheduler, NothingScheduler
from .config import TrainConfig
from .saving import ModelSavingStrategy, get_saving_callback
from .dataset.util import DatasetConfig
from .dataloader import get_dataloader
from .utils.logging import get_trackers


class ModelForTraining(ABC):
    accelerator: Accelerator
    config: TrainConfig
    model_config: BaseModel
    model_config_class: type[BaseModel]

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    _current_step: int = 0
    _logs_at_step: dict = {}
    _logs_at_epoch: dict[str, list] = {}

    def __init__(
        self,
        accelerator: Accelerator,
        config: TrainConfig,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config
        self.accelerator = accelerator

        self.validate_config()

    def validate_config(self):
        self.model_config = self.model_config_class.model_validate(self.config.model)

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def sanity_check(self):
        pass

    def setup_optimizer(self):
        optimizer = get_optimizer(
            self.config.optimizer.name,
            self.model.parameters(),
            **self.config.optimizer.args,
        )
        if (scheduler_config := self.config.scheduler) is not None:
            scheduler = get_scheduler(
                optimizer,
                scheduler_config.name,
                **scheduler_config.args,
            )
        else:
            scheduler = NothingScheduler(optimizer)

        self.optimizer = self.accelerator.prepare_optimizer(
            optimizer,
        )  # type: ignore  # Accelerator's prepare_optimizer method may not be recognized by type checkers
        self.scheduler = scheduler

    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(self, batch) -> torch.Tensor:
        pass

    def backward(self, loss: torch.Tensor):
        self.before_backward()

        self.accelerator.backward(loss)

        self.after_backward()

    @abstractmethod
    def before_load_model(self):
        pass

    @abstractmethod
    def after_load_model(self):
        pass

    def before_train_step(self):
        self.optimizer.zero_grad()
        self.increment_step()

    def after_train_step(self):
        self._log_metadata()
        self._send_logs_at_step()

    @abstractmethod
    def before_eval_step(self):
        pass

    def after_eval_step(self):
        self._send_logs_at_step()

    @abstractmethod
    def before_backward(self):
        pass

    def after_backward(self):
        self.optimizer.step()
        self.scheduler.step()

    def before_train_epoch(self):
        self.model.train()
        # if the optimizer has train(), call it
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()  # type: ignore  # Some optimizers might not have a train method

    def after_train_epoch(self):
        self._send_logs_at_epoch()
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()  # type: ignore  # Some optimizers might not have an eval method

    def before_eval_epoch(self):
        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()  # type: ignore

    def after_eval_epoch(self):
        self._send_logs_at_epoch()

    def before_save_model(self):
        pass

    def after_save_model(self):
        pass

    def print(self, *args, **kwargs):
        self.accelerator.print(*args, **kwargs)

    def log(
        self,
        name: str,
        value,
        on_step: bool = True,
        on_epoch: bool = False,
    ):
        if isinstance(value, torch.Tensor):
            with torch.no_grad():
                value = self.accelerator.gather(value)
                assert isinstance(value, torch.Tensor)
                value = value.mean().item()

        if on_step:
            self._logs_at_step[name] = value
        if on_epoch:
            if name not in self._logs_at_epoch:
                self._logs_at_epoch[name] = []
            self._logs_at_epoch[name].append(value)

    def _send_logs_at_step(self):
        self.accelerator.log(self._logs_at_step, step=self._current_step)
        self._logs_at_step = {}

    def _send_logs_at_epoch(self):
        for name, values in self._logs_at_epoch.items():
            if isinstance(values[0], torch.Tensor):
                values = [v.mean().item() for v in values]

            if isinstance(values[0], float) or isinstance(values[0], int):
                self.accelerator.log(
                    {f"{name}_epoch": sum(values) / len(values)},
                    step=self._current_step,
                )
            else:
                for i, value in enumerate(values):
                    self.accelerator.log(
                        {f"{name}_{i}_epoch": value}, step=self._current_step
                    )
        self._logs_at_epoch = {}

    def _log_metadata(self):
        # learning rate
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr/group_{i}", param_group["lr"], on_step=True, on_epoch=False)

    def increment_step(self):
        self._current_step += 1


class Trainer:
    model: ModelForTraining

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
        train_ds, eval_ds = self.dataset_config.get_dataset()

        train_dataloader = get_dataloader(
            train_ds,
            batch_size=self.dataset_config.batch_size,
            shuffle=self.dataset_config.shuffle,
            num_workers=self.dataset_config.num_workers,
        )
        eval_dataloader = get_dataloader(
            eval_ds,
            batch_size=self.dataset_config.batch_size,
            shuffle=False,
            num_workers=self.dataset_config.num_workers,
        )
        self.train_dataloader, self.eval_dataloader = (
            self.accelerator.prepare_data_loader(train_dataloader, eval_dataloader)
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

    def before_train(self):
        self.print("before_train()")
        self.print(f"Seed: {self.seed}")
        set_seed(self.seed)

        self.print("Setting up dataloaders")
        self.prepare_dataloaders()

        self.print("Setting up saving strategy")
        self.prepare_saving_strategy()

        self.print("Setting up model")
        self.model.setup_model()
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

                # To avoid saving twice at the end of an epoch,
                # skip the saving check during the epoch and save only at the end.
                if steps != len(self.train_dataloader) - 1:
                    self.call_saving_callbacks(epoch, current_step)

                if self.only_sanity_check:
                    break

            self.model.after_train_epoch()
            self.model.log("epoch", epoch)
            self.call_saving_callbacks(epoch, current_step)

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
