from safetensors.torch import save_file
from torch.nn.modules import Module

from .util import ModelSavingCallback, ModelSavingCallbackConfig


class SafetensorsSavingCallbackConfig(ModelSavingCallbackConfig):
    type: str = "safetensors"


class SafetensorsSavingCallback(ModelSavingCallback):
    def save(self, model: Module, epoch: int, steps: int, metadata: dict | None = None):
        file_name = self.format_template(name=self.name, epoch=epoch, steps=steps)
        save_path = self.save_dir / file_name

        if (parent_dir := save_path.parent) and not parent_dir.exists():
            parent_dir.mkdir(parents=True)

        save_file(model.state_dict(), self.save_dir / file_name, metadata)

        return save_path
