import yaml
from pathlib import Path
from pydantic import BaseModel
import json

import torch.utils.data as data
from datasets import Dataset


from ..util import DatasetConfig, HFDatasetWrapper


class T2IPreviewArgs(BaseModel):
    prompt: str
    negative_prompt: str | None = ""
    height: int = 1024
    width: int = 1024
    cfg_scale: float = 5.0
    num_steps: int = 20

    seed: int = 0


class TextToImagePreviewConfig(DatasetConfig):
    path: str

    def get_preview_args(self) -> list[T2IPreviewArgs]:
        path = Path(self.path)
        assert path.exists()

        extension = path.suffix.lower()
        if extension == ".yaml" or extension == ".yml":
            with open(self.path, "r") as f:
                config = yaml.safe_load(f)
            return [T2IPreviewArgs.model_validate(item) for item in config]

        elif extension == ".json":
            with open(self.path, "r") as f:
                config = json.load(f)
            return [T2IPreviewArgs.model_validate(item) for item in config]

        raise ValueError(f"Unknown extension: {extension}")

    def get_dataset(self) -> data.Dataset:
        args = self.get_preview_args()

        ds = Dataset.from_generator(
            self._generate_row,
            gen_kwargs={"items": args},
        )
        assert isinstance(ds, Dataset)

        return HFDatasetWrapper(ds)

    def _generate_row(self, items: list[T2IPreviewArgs]):
        for item in items:
            yield {
                "prompt": item.prompt,
                "negative_prompt": item.negative_prompt,
                "height": item.height,
                "width": item.width,
                "cfg_scale": item.cfg_scale,
                "num_steps": item.num_steps,
                "seed": item.seed,
            }
