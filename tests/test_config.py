from src.config import TrainConfig


def test_validate_config():
    path = "tests/assets/debug_dataset.yml"

    config = TrainConfig.from_config_file(path)
    assert config is not None

    assert config.trainer is not None
    assert config.trainer.debug_mode == "dataset"
