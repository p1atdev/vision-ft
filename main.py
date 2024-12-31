import click

from src.models.auraflow import AuraFlowForTraining
from src.trainer.common import Trainer
from src.config import TrainConfig
from src.dataset.text_to_image import TextToImageDatasetConfig


@click.command()
@click.option("--config", type=str, required=True)
def main(config: str):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
    )
    trainer.register_dataset_class(TextToImageDatasetConfig)
    trainer.register_model_class(AuraFlowForTraining)

    trainer.train()


if __name__ == "__main__":
    main()
