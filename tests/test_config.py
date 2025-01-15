from src.config import TrainConfig
from src.modules.peft import LoRAConfig


def test_validate_config():
    path = "tests/assets/debug_dataset.yml"

    config = TrainConfig.from_config_file(path)
    assert config is not None

    assert config.trainer is not None
    assert config.trainer.debug_mode == "dataset"

    target_config = config.peft
    assert target_config is not None
    assert isinstance(target_config, list)
    if (peft_config := target_config[0].config) and isinstance(peft_config, LoRAConfig):
        assert peft_config.type == "lora"
        assert peft_config.rank == 8
    else:
        assert False, "Unexpected config type"
