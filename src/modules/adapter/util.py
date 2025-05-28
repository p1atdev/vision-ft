from abc import ABC, abstractmethod
from pydantic import BaseModel

import torch
from torch import nn

from accelerate import init_empty_weights


from ...utils.state_dict import RegexMatch


class Adapter(nn.Module, ABC):
    """
    Abstract base class for adapters.
    """

    target_key: RegexMatch

    @classmethod
    def from_module(cls, module: nn.Module, *args, **kwargs):
        """
        Initializes the adapter from a given module.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def get_module_dict(self) -> dict[str, nn.Module]:
        """
        Returns a dictionary of modules that this adapter manages.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AdapterManager(nn.Module, ABC):
    module_dict: nn.ModuleDict

    adapter_class: type[Adapter]
    adapter_config: BaseModel

    def __init__(self, adapter_class: type[Adapter], adapter_config: BaseModel):
        super().__init__()

        self.module_dict = nn.ModuleDict({})

        self.adapter_class = adapter_class
        self.adapter_config = adapter_config

    @abstractmethod
    def apply_adapter(self, model: nn.Module):
        """
        Applies the specified adapter to the model.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load_adapter(self, model: nn.Module, state_dict: dict[str, torch.Tensor]):
        with init_empty_weights():
            self.model.apply_adapter(model)

        # load state dict
        # rename
        state_dict = {k.replace(".", "!"): v for k, v in state_dict.items()}
        self.module_dict.load_state_dict(state_dict, assign=True)

    def parameters(self):
        """
        Returns the parameters of the adapter manager.
        """
        assert hasattr(self, "module_dict"), "Adapter manager not initialized."
        assert isinstance(self.module_dict, nn.ModuleDict), (
            "Adapter manager not initialized."
        )
        assert self.module_dict is not None, "Adapter manager not initialized."
        assert len(self.module_dict) > 0, "Adapter manager not initialized."

        return self.module_dict.parameters()

    def get_state_dict(self):
        """
        Returns the state dictionary of the adapter manager.
        """
        return self.module_dict.state_dict()
