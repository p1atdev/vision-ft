from pathlib import Path
from abc import ABC, abstractmethod
from pydantic import BaseModel

import torch
import torch.nn as nn


class ModelSavingStrategyConfig(BaseModel):
    per_epochs: int | float | None = 1
    per_steps: int | None = None
    save_last: bool = True


class ModelSavingStrategy:
    per_epochs: int | float | None = None
    per_steps: int | None = None
    save_last: bool = True

    total_epochs: int
    steps_per_epoch: int

    def __init__(
        self,
        total_epochs: int,
        steps_per_epoch: int,
        per_epochs: int | float | None,
        per_steps: int | None,
        save_last: bool,
    ):
        self.per_epochs = per_epochs
        self.per_steps = per_steps
        self.save_last = save_last

        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch

        self.sanity_check()

    @classmethod
    def from_config(
        cls, config: ModelSavingStrategyConfig, total_epochs: int, steps_per_epoch: int
    ) -> "ModelSavingStrategy":
        return cls(
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch,
            **config.model_dump(),
        )

    @property
    def total_steps(self) -> int:
        return self.total_epochs * self.steps_per_epoch

    def check_strategy(self) -> bool:
        if self.per_epochs is None and self.per_steps is None:
            return True

        if self.per_epochs is not None:
            if self.per_epochs <= 0:
                raise ValueError("per_epochs must be greater than 0")

            if isinstance(self.per_epochs, float):
                if self.per_epochs >= 1:
                    raise ValueError("per_epochs must be less than 1 if float")

                if self.per_steps is not None:
                    raise ValueError("per_epochs and per_steps cannot be set together")

            elif isinstance(self.per_epochs, int):
                if self.per_epochs > self.total_epochs:
                    raise ValueError(
                        "per_epochs must be less than or equal to total_epochs"
                    )

        if self.per_steps is not None:
            if self.per_steps <= 0:
                raise ValueError("per_steps must be greater than 0")

            if self.per_steps > self.total_steps:
                raise ValueError("per_steps must be less than or equal to total_steps")

        return True

    def sanity_check(self):
        self.check_strategy()

    @property
    def _per_epochs(self) -> int | None:
        if self.per_epochs is None:
            return None

        if isinstance(self.per_epochs, int):
            return self.per_epochs

        if isinstance(self.per_epochs, float):
            return None

        raise ValueError("per_epochs must be int or float")

    @property
    def _per_steps(self) -> int | None:
        if isinstance(self.per_epochs, float):
            return int(self.per_epochs * self.steps_per_epoch)

        return self.per_steps

    def should_save(self, epoch: int, steps: int) -> bool:
        # saving is disabled
        if self.steps_per_epoch is None and self.total_epochs is None:
            return False

        if epoch == 0 and steps == 0:
            return False  # skip the first step

        if self._per_epochs is not None and epoch != 0:
            if steps % (self.steps_per_epoch * self._per_epochs) == 0:
                return True

        if self._per_steps is not None and steps != 0:
            if steps % self._per_steps == 0:
                return True

        return False


class ModelSavingCallbackConfig(BaseModel):
    type: str

    name: str
    save_dir: str | Path


class ModelSavingCallback(ABC):
    save_name_template: str = "{name}_{epoch:05}e_{steps:06}s.safetensors"
    name: str
    _save_dir: Path

    def __init__(
        self,
        name: str,
        save_dir: str | Path,
        save_name_template: str | None = None,
    ) -> None:
        super().__init__()

        self.name = name
        self._save_dir = save_dir if isinstance(save_dir, Path) else Path(save_dir)
        if save_name_template is not None:
            self.save_name_template = save_name_template

        self.sanity_check()

    @classmethod
    def from_config(
        cls, config: ModelSavingCallbackConfig, **kwargs
    ) -> "ModelSavingCallback":
        config_dict = config.model_dump()
        config_dict.pop("type")

        return cls(**config_dict, **kwargs)

    def sanity_check(self):
        pass

    def format_template(self, **kwargs) -> str:
        return self.save_name_template.format(**kwargs)

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @abstractmethod
    def save(
        self, model: nn.Module, epoch: int, steps: int, metadata: dict | None = None
    ):
        pass
