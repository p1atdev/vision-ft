from src.config import TrainConfig
from src.modules.peft import LoRAConfig


def test_validate_config():
    path = "tests/assets/debug_dataset.yml"

    config = TrainConfig.from_config_file(path)
    assert config is not None

    assert config.trainer is not None
    assert config.trainer.debug_mode == "dataset"

    peft = config.peft
    assert peft is not None
    assert peft.type == "lora"

    assert isinstance(peft, LoRAConfig)
    assert peft.rank == 8
