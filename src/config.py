from typing import Literal, Union

import yaml
from pathlib import Path

from pydantic import BaseModel, field_validator

from .saving import (
    ModelSavingCallbackConfgiAlias,
    SafetensorsSavingCallbackConfig,
    ModelSavingStrategyConfig,
)
from .preview import (
    PreviewCallbackConfigAlias,
    PreviewStrategyConfig,
    LocalPreviewCallbackConfig,
)
from .modules.peft import PeftTargetConfig
from .dataset import PreviewDatasetAlias


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
    callbacks: list[ModelSavingCallbackConfgiAlias] = [
        SafetensorsSavingCallbackConfig(name="model", save_dir="./output")
    ]

    rename_key_map: dict[str, str] = {}


class PreviewConfig(BaseModel):
    strategy: PreviewStrategyConfig = PreviewStrategyConfig()
    callbacks: list[PreviewCallbackConfigAlias] = [
        LocalPreviewCallbackConfig(save_dir="./output/preview")
    ]

    data: PreviewDatasetAlias


class TrackerConfig(BaseModel):
    project_name: str
    loggers: list[Literal["wandb", "tensorboard"]]


DEBUG_MODE_TYPE = Literal[
    False,  # not debug mode
    "sanity_check",  # check model sanity
    "1step",  # pass only 1 step
    "dataset",  # check dataset
]


class TrainerConfig(BaseModel):
    debug_mode: DEBUG_MODE_TYPE = False

    torch_compile: bool = False
    torch_compile_args: dict = {}

    gradient_checkpointing: bool = False
    gradient_accumulation_steps: int = 1

    clip_grad_norm: float | None = None
    clip_grad_value: float | None = None

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    fp32_matmul_precision: Literal["highest", "high", "medium"] | None = None
    # https://pytorch.org/docs/stable/notes/cuda.html
    allow_tf32: bool = False


class TrainConfig(BaseModel):
    model: dict | BaseModel
    dataset: dict | BaseModel
    peft: PeftTargetConfig | list[PeftTargetConfig] | None = None

    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig | None = None
    saving: SavingConfig | None = SavingConfig()
    preview: PreviewConfig | None = None
    tracker: TrackerConfig | None = None
    trainer: TrainerConfig = TrainerConfig()

    seed: int = 42

    num_train_epochs: int = 1

    def to_dict(self) -> dict:
        return self.model_dump()

    def ve_to(self, dir: Path | str, filename: str = "config.yaml"):
        if isinstance(dir, str):
            dir = Path(dir)

        dir.mkdir(parents=True, exist_ok=True)
        with open(dir / filename, "w") as f:
            yaml.dump(self.to_dict(), f)

    @staticmethod
    def from_config_file(path: str) -> "TrainConfig":
        with open(path, "r") as f:
            config = yaml.safe_load(f)

        return TrainConfig.model_validate(config, strict=True)
