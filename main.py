import click

from src.models.auraflow import AuraFlowForTraining
from src.trainer.t2i import Trainer
from src.config import TrainConfig
from src.dataset.t2i import T2IDatasetConfig


@click.command()
@click.option("--config", type=str, required=True)
@click.option("--debug", is_flag=True)
def main(config: str, debug: bool):
    _config = TrainConfig.from_config_file(config)

    trainer = Trainer(
        _config,
        only_sanity_check=debug,
    )
    trainer.register_dataset_class(T2IDatasetConfig)
    trainer.register_model_class(AuraFlowForTraining)

    trainer.train()


if __name__ == "__main__":
    main()
