from typing import Literal
from PIL import Image
from pydantic import BaseModel

import torch
from torch import nn

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotImageClassification,
    BatchEncoding,
)

from .utils import RewardModelMixin, RewardModelConfig


class PickScoreConfig(RewardModelConfig):
    type: Literal["pickscore"] = "pickscore"

    model_id: str = "yuvalkirstain/PickScore_v1"

    def load_model(self, device: torch.device) -> "PickScoreRewardModel":
        """
        Load the PickScore model on the specified device.
        """
        return PickScoreRewardModel(device=device, model_id=self.model_id)


# https://huggingface.co/yuvalkirstain/PickScore_v1
class PickScoreRewardModel(RewardModelMixin):
    def __init__(
        self,
        device: torch.device,
        model_id: str = "yuvalkirstain/PickScore_v1",
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotImageClassification.from_pretrained(model_id)
        self.model.to(device)
        self.model.eval()

    def preprocess(
        self, images: list[Image.Image], prompts: list[str]
    ) -> tuple[BatchEncoding, BatchEncoding]:
        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

        return image_inputs, text_inputs

    @torch.no_grad()
    def __call__(
        self,
        images: list[Image.Image],
        prompts: list[str],
    ) -> torch.Tensor:
        image_inputs, text_inputs = self.preprocess(images, prompts)

        image_embs = self.model.get_image_features(**image_inputs.to(self.model.device))
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs.to(self.model.device))
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)

        # score
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)

        return probs
