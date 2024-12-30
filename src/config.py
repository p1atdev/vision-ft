from typing import Literal

import yaml
from pathlib import Path

from pydantic import BaseModel

from .saving import (
    HFHubSavingCallbackConfig,
    SafetensorsSavingCallbackConfig,
    ModelSavingStrategyConfig,
)
from .modules.peft import PeftConfigMixin


class OptimizerConfig(BaseModel):
    name: str = "torch.optim.AdamW"
    args: dict = {
        "lr": 1e-3,
    }


class SchedulerConfig(BaseModel):
    name: str = "torch.optim.lr_scheduler.ConstantLR"
    args: dict = {}


class SavingConfig(BaseModel):
    strategy: ModelSavingStrategyConfig = ModelSavingStrategyConfig()
    callbacks: list[SafetensorsSavingCallbackConfig | HFHubSavingCallbackConfig] = [
        SafetensorsSavingCallbackConfig(name="model", save_dir="./output")
    ]


class TrackerConfig(BaseModel):
    name: Literal["wandb", "tensorboard", "csv"]
    args: dict = {}


class TrainConfig(BaseModel):
    model: dict | BaseModel
    dataset: dict | BaseModel
    peft: PeftConfigMixin | None = None

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig | None = None
    saving: SavingConfig | None = SavingConfig()
    trackers: list[TrackerConfig] | None = []

    seed: int = 42

    num_train_epochs: int = 1

    torch_compile: bool = False
    fp32_matmul_precision: Literal["highest", "high", "medium"] | None = None

    def to_dict(self) -> dict:
        return self.model_dump()

    def save_to(self, dir: Path | str, filename: str = "config.yaml"):
        if isinstance(dir, str):
            dir = Path(dir)

        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_config_file(path: str) -> "TrainConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return TrainConfig.model_validate(config)
