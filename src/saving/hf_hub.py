from pathlib import Path
from safetensors.torch import save_file
from huggingface_hub import HfApi

from torch._tensor import Tensor
from torch.nn.modules import Module

from .safetensors import SafetensorsSavingCallback, SafetensorsSavingCallbackConfig


class HFHubSavingCallbackConfig(SafetensorsSavingCallbackConfig):
    type: str = "hf_hub"

    hub_id: str
    dir_in_repo: str
    repo_type: str = "model"


class HFHubSavingCallback(SafetensorsSavingCallback):
    api = HfApi()

    hub_id: str
    dir_in_repo: str
    repo_type: str = "model"

    def __init__(
        self,
        hub_id: str,
        dir_in_repo: str = "",
        repo_type: str = "model",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.hub_id = hub_id
        self.dir_in_repo = dir_in_repo
        self.repo_type = repo_type

    def save(self, model: Module, epoch: int, steps: int, metadata: dict | None = None):
        state_dict = model.state_dict()

        return self.save_state_dict(state_dict, epoch, steps, metadata)

    def save_state_dict(
        self,
        state_dict: dict[str, Tensor],
        epoch: int,
        steps: int,
        metadata: dict | None = None,
    ):
        save_path = super().save_state_dict(state_dict, epoch, steps, metadata)
        filename = save_path.name

        self.api.upload_file(
            path_or_fileobj=save_path,
            repo_id=self.hub_id,
            path_in_repo=f"{self.dir_in_repo}/{filename}",
            repo_type=self.repo_type,
            commit_message=f"Upload model at epoch {epoch} and steps {steps}",
        )

        return save_path
