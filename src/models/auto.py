from abc import ABC, abstractmethod
from typing import Literal
from pydantic import BaseModel

import torch
import torch.nn as nn

import timm
from transformers import AutoModel, AutoConfig
from transformers.activations import ACT2FN


class AbstractAutoModelConfig(BaseModel, ABC):
    type: str
    model_name: str
    config: dict

    @abstractmethod
    def setup_model(self) -> nn.Module:
        """
        Returns the model instance.
        """
        pass

    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        Loads the model instance.
        """
        pass


class TransformersModelConfig(AbstractAutoModelConfig):
    type: Literal["transformers"] = "transformers"
    model_name: str
    config: dict = {}
    pretrained: bool = True

    def setup_model(self) -> nn.Module:
        """
        Returns the pretrained model instance.
        """

        config = AutoConfig.from_pretrained(self.model_name, **self.config)
        model = AutoModel.from_config(config)
        return model

    def load_model(self) -> nn.Module:
        """
        Loads the pretrained model instance.
        """
        if self.pretrained:
            return AutoModel.from_pretrained(self.model_name, **self.config)

        model = AutoModel.from_config(
            AutoConfig.from_pretrained(self.model_name, **self.config)
        )

        return model


class TimmModelConfig(AbstractAutoModelConfig):
    type: Literal["timm"] = "timm"
    model_name: str
    config: dict = {}
    pretrained: bool = True

    def setup_model(self) -> nn.Module:
        """
        Returns the pretrained model instance.
        """

        self.config["num_classes"] = 0  # Remove the classification head

        model = timm.create_model(self.model_name, pretrained=False, **self.config)
        return model

    def load_model(self) -> nn.Module:
        """
        Loads the pretrained model instance.
        """
        if self.pretrained:
            return timm.create_model(self.model_name, pretrained=True, **self.config)

        model = timm.create_model(self.model_name, pretrained=False, **self.config)

        return model


AutoModelConfig = TransformersModelConfig | TimmModelConfig


class AutoImageEncoder(nn.Module):
    def __init__(
        self,
        config: AutoModelConfig,
    ):
        super().__init__()

        self.config = config
        self.model = config.setup_model()

    def _load_model(self):
        self.model = self.config.load_model()

    def encode(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(self.config, TransformersModelConfig):
            outputs = self.model(pixel_values, **kwargs)
            return outputs.pooler_output
        elif isinstance(self.config, TimmModelConfig):
            last_hidden_state = self.model.forward_features(pixel_values, **kwargs)
            pooler_output = self.model.forward_head(last_hidden_state)
            return pooler_output
        else:
            raise NotImplementedError(
                f"Model type {type(self.model)} is not supported."
            )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode(pixel_values, **kwargs)
