import torch.nn as nn

from src.saving.util import ModelSavingStrategy
from src.saving import SafetensorsSavingCallback, HFHubSavingCallback


def test_model_saving_strategy_sanity_check():
    total_epochs = 200
    steps_per_epoch = 128

    test_cases = [
        (
            {
                "per_epochs": 1,
                "per_steps": None,
            },
            "ok",
            1,
            None,
        ),
        (
            {
                "per_epochs": None,
                "per_steps": 10,
            },
            "ok",
            None,
            10,
        ),
        (
            {
                "per_epochs": 0.1,
                "per_steps": 10,
            },
            "throw",
            None,
            None,
        ),
        (
            {
                "per_epochs": 0.1,
                "per_steps": None,
            },
            "ok",
            None,
            12,
        ),
        (
            {
                "per_epochs": 2,
                "per_steps": 10,
            },
            "ok",
            2,
            10,
        ),
    ]

    for case in test_cases:
        result = "ok"

        try:
            strategy = ModelSavingStrategy(
                total_epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                save_last=True,
                **case[0],
            )
            strategy.sanity_check()

            assert strategy._per_epochs == case[2], case
            assert strategy._per_steps == case[3], case
        except ValueError:
            result = "throw"

        assert result == case[1], case


def test_model_saving_strategy_should_save():
    test_cases = [
        (
            {
                "per_epochs": 1,
                "per_steps": None,
            },
            0,  # epoch
            0,  # steps
            False,
        ),
        (
            {
                "per_epochs": None,
                "per_steps": 10,
            },
            0,
            100,
            True,
        ),
        (
            {
                "per_epochs": 0.1,
                "per_steps": 10,
            },
            1,
            100,
            "throw",
        ),
        (
            {
                "per_epochs": 0.1,
                "per_steps": None,
            },
            1,
            100,
            True,
        ),
        (
            {
                "per_epochs": 2,
                "per_steps": 10,
            },
            2,
            100,
            True,
        ),
        (
            {
                "per_epochs": 3,
                "per_steps": 100,
            },
            5,
            150,
            False,
        ),
    ]

    for case in test_cases:
        result = None

        try:
            strategy = ModelSavingStrategy(
                total_epochs=100,
                steps_per_epoch=1000,
                save_last=True,
                **case[0],
            )
            strategy.sanity_check()
            result = strategy.should_save(epoch=case[1], steps=case[2])
        except ValueError:
            result = "throw"

        assert result == case[3], case


def test_safetensors_saving_callback():
    callback = SafetensorsSavingCallback(
        name="test",
        save_dir="./output",
    )

    model = nn.Transformer()

    try:
        save_path = callback.save(model, 0, 1, {"test": "test1"})
        assert save_path.exists()
    except Exception as e:
        assert False, e


#! requires huggingface WRITE token
def test_hfhub_saving_callback():
    callback = HFHubSavingCallback(
        name="test",
        hub_id="p1atdev/test",
        dir_in_repo="test-model",
        save_dir="./output",
    )

    model = nn.Transformer()

    try:
        save_path = callback.save(model, 0, 2, {"test": "test1"})
        assert save_path.exists()
    except Exception as e:
        assert False, e
