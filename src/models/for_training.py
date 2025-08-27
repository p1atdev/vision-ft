from abc import ABC, abstractmethod
from typing import Any
from pydantic import BaseModel
from PIL import Image

import torch
import torch.nn as nn

from accelerate import Accelerator

from ..config import TrainConfig


class ModelForTraining(ABC, nn.Module):
    accelerator: Accelerator
    config: TrainConfig
    model_config: BaseModel
    model_config_class: type[BaseModel]

    model: nn.Module

    _current_step: int = 0
    _logs_at_step: dict = {}
    _logs_at_epoch: dict[str, list] = {}
    _is_peft: bool = False

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

    def _set_is_peft(self, is_peft: bool):
        self._is_peft = is_peft

    def load_peft_weights(self):
        pass

    @abstractmethod
    def before_setup_model(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def after_setup_model(self):
        if self.config.trainer.torch_compile:
            self.print("torch.compile is enabled")
            self.model = torch.compile(
                self.model,
                **self.config.trainer.torch_compile_args,
            )  # type: ignore

        self.accelerator.wait_for_everyone()

    @abstractmethod
    def sanity_check(self):
        pass

    @abstractmethod
    def train_step(self, batch) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(self, batch) -> torch.Tensor:
        pass

    def before_train_step(self):
        self.increment_step()

    def after_train_step(self):
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
        if self.accelerator.sync_gradients:
            if self.config.trainer.clip_grad_norm is not None:
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.clip_grad_norm,
                )
            if self.config.trainer.clip_grad_value is not None:
                self.accelerator.clip_grad_value_(
                    self.model.parameters(),
                    self.config.trainer.clip_grad_value,
                )

    def before_train_epoch(self):
        self.model.train()

    def after_train_epoch(self):
        self._send_logs_at_epoch()
        self.model.eval()

    def before_eval_epoch(self):
        self.model.eval()

    def after_eval_epoch(self):
        self._send_logs_at_epoch()

    def get_state_dict_to_save(
        self,
    ) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def get_metadata_to_save(
        self,
    ) -> dict[str, str]:
        # additional metadata to save with weights into the safetensors
        return {}

    def before_save_model(self):
        pass

    def after_save_model(self):
        pass

    def before_preview(self):
        pass

    def before_preview_step(
        self,
    ):
        pass

    @abstractmethod
    def preview_step(
        self,
        batch,
        preview_index: int,
    ) -> Any:
        """
        e.g.) generate sample images for checking the training progress
        """
        pass

    def after_preview_step(
        self,
    ):
        pass

    def after_preview(self):
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

    def increment_step(self):
        self._current_step += 1

    def forward(self, *args, **kwargs):
        return self.train_step(*args, **kwargs)
