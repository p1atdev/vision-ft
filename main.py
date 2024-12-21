import argparse

from src.models.mnist import MnistModelForTraining
from src.trainer import Trainer
from src.config import TrainConfig
from src.dataset.mnist import MnistDatasetConfig


def prepare_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = prepare_args()

    config = TrainConfig.from_config_file(args.config)

    trainer = Trainer(
        config,
        only_sanity_check=args.debug,
    )
    trainer.register_dataset_class(MnistDatasetConfig)
    trainer.register_model_class(MnistModelForTraining)

    trainer.train()


if __name__ == "__main__":
    main()
