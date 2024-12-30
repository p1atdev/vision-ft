from pydantic import BaseModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as metrics

from ..trainer.t2i import ModelForTraining


class MnistConfig(BaseModel):
    num_pixels: int = 784
    hidden_dim: int = 128
    num_labels: int = 10


class MnistModel(nn.Module):
    def __init__(self, config: MnistConfig):
        super().__init__()

        self.config = config

        self.layers = nn.Sequential(
            nn.Linear(config.num_pixels, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_labels),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, pixel_values: torch.Tensor):
        # reshape
        h = pixel_values.view(-1, 784)
        logits = self.layers(h)

        return logits


class MnistModelForTraining(ModelForTraining, nn.Module):
    model: nn.Module

    model_config: MnistConfig
    model_config_class = MnistConfig

    def setup_model(self):
        with self.accelerator.main_process_first():
            print("Initializing model")

            model = MnistModel(self.model_config)

        self.accelerator.wait_for_everyone()

        self.model = self.accelerator.prepare_model(model)

    @torch.no_grad()
    def sanity_check(self):
        with self.accelerator.autocast():
            pixel_values = torch.randn(1, self.model_config.num_pixels)
            logits = self.model(pixel_values.to(self.accelerator.device))
            assert logits.shape == (1, self.model_config.num_labels)

    def log_metrics(self, logits: torch.Tensor, targets: torch.Tensor):
        preds = torch.argmax(logits, dim=1)
        targets = targets.squeeze()
        accuracy = metrics.accuracy(
            preds, targets, task="multiclass", num_classes=self.model_config.num_labels
        )
        precision = metrics.precision(
            preds, targets, task="multiclass", num_classes=self.model_config.num_labels
        )
        recall = metrics.recall(
            preds, targets, task="multiclass", num_classes=self.model_config.num_labels
        )
        f1 = metrics.f1_score(
            preds, targets, task="multiclass", num_classes=self.model_config.num_labels
        )

        for name, value in {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }.items():
            self.log(f"eval/{name}", value, on_step=False, on_epoch=True)

    def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        pixel_values, targets = batch

        logits = self.model(pixel_values)
        loss = F.cross_entropy(logits, targets.squeeze())

        self.log("train/loss", loss, on_step=True, on_epoch=True)

        return loss

    def eval_step(self, batch: tuple[torch.Tensor, torch.Tensor]):
        pixel_values, targets = batch

        logits = self.model(pixel_values)
        loss = F.cross_entropy(logits, targets.squeeze())

        self.log("eval/loss", loss, on_step=False, on_epoch=True)
        self.log_metrics(logits, targets)

        return loss

    def before_load_model(self):
        super().before_load_model()

    def after_load_model(self):
        super().after_load_model()

    def before_eval_step(self):
        super().before_eval_step()

    def before_backward(self):
        super().before_backward()
