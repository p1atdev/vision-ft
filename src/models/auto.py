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
    def get_model(self) -> nn.Module:
        """
        Returns the model instance.
        """
        pass


class TransformersModelConfig(AbstractAutoModelConfig):
    type: Literal["transformers"] = "transformers"
    model_name: str
    config: dict = {}
    pretrained: bool = True

    def get_model(self) -> nn.Module:
        """
        Returns the pretrained model instance.
        """
        if self.pretrained:
            model = AutoModel.from_pretrained(self.model_name, **self.config)
            return model

        config = AutoConfig.from_pretrained(self.model_name, **self.config)
        model = AutoModel.from_config(config)
        return model


class TimmModelConfig(AbstractAutoModelConfig):
    type: Literal["timm"] = "timm"
    model_name: str
    config: dict = {}
    pretrained: bool = True

    def get_model(self) -> nn.Module:
        """
        Returns the pretrained model instance.
        """

        self.config["num_classes"] = 0  # Remove the classification head

        if self.pretrained:
            model = timm.create_model(self.model_name, pretrained=True, **self.config)
            return model

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
        self.model = config.get_model()

    def encode(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(self.model, TransformersModelConfig):
            outputs = self.model(pixel_values, **kwargs)
            return outputs.pooler_output
        elif isinstance(self.model, TimmModelConfig):
            last_hidden_state = self.model.forward_features(pixel_values, **kwargs)
            pooler_output = self.model.forward_head(last_hidden_state)
            return pooler_output
        else:
            raise NotImplementedError(
                f"Model type {type(self.model)} is not supported."
            )

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode(pixel_values, **kwargs)
