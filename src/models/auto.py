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

    feature_type: Literal["hidden_state", "pooler_output"] = "pooler_output"
    hidden_state_index: int = -1

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

        model = timm.create_model(self.model_name, pretrained=False, **self.config)
        model.reset_classifier(0)  # remove classifier head

        return model

    def load_model(self) -> nn.Module:
        """
        Loads the pretrained model instance.
        """
        if self.pretrained:
            model = timm.create_model(self.model_name, pretrained=True, **self.config)
        else:
            model = timm.create_model(self.model_name, pretrained=False, **self.config)

        model.reset_classifier(0)  # remove classifier head

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
        feature_type = self.config.feature_type

        if isinstance(self.config, TransformersModelConfig):
            outputs = self.model(pixel_values, return_hidden_states=True, **kwargs)
            if feature_type == "hidden_state":
                # Get the last hidden state
                hidden_state = outputs.hidden_states[self.config.hidden_state_index]
                # maybe [batch_size, NxN, num_features]?

                return hidden_state

            elif feature_type == "pooler_output":
                return outputs.pooler_output

        elif isinstance(self.config, TimmModelConfig):
            if feature_type == "hidden_state":
                _output, intermediates = self.model.forward_intermediates(
                    pixel_values, **kwargs
                )
                hidden_state = intermediates[self.config.hidden_state_index]
                # maybe [batch_size, num_features, N, N]
                batch_size, dim, _h, _w = hidden_state.size()

                hidden_state = hidden_state.permute(0, 2, 3, 1).reshape(
                    batch_size, -1, dim
                )
                return hidden_state

            elif feature_type == "pooler_output":
                last_hidden_state = self.model.forward_features(pixel_values, **kwargs)
                pooler_output = self.model.forward_head(last_hidden_state)
                return pooler_output

        raise NotImplementedError(f"Model type {type(self.model)} is not supported.")

    def forward(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.encode(pixel_values, **kwargs)
