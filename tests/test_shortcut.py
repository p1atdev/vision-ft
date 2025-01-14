import torch

from src.modules.loss.shortcut import (
    prepare_self_consistency_targets,
    prepare_random_shortcut_durations,
)


def test_random_shortcut_duration():
    batch_size = 4

    inference_steps, shortcut_exponent, shortcut_duration, departure_timesteps = (
        prepare_random_shortcut_durations(
            batch_size=batch_size,
            max_pow=7,
        )
    )

    assert inference_steps.shape == (batch_size,)
    assert shortcut_exponent.shape == (batch_size,)
    assert shortcut_duration.shape == (batch_size,)
    assert departure_timesteps.shape == (batch_size,)


def test_random_shortcut_duration_timesteps():
    torch.manual_seed(42)

    for _ in range(100):
        batch_size = 8

        inference_steps, shortcut_exponent, shortcut_duration, departure_timesteps = (
            prepare_random_shortcut_durations(
                batch_size=batch_size,
                min_pow=0,
                max_pow=3,
            )
        )

        assert (inference_steps >= 1).all()
        assert (inference_steps <= 2**3).all()

        assert (shortcut_exponent >= 0).all()
        assert (shortcut_exponent <= 3).all()

        assert (departure_timesteps > 0).all()
        assert (departure_timesteps <= 1).all()  # must be before fully denoised

        assert (departure_timesteps - shortcut_duration < 1).all()
        assert (departure_timesteps - shortcut_duration >= 0).all()
        assert (departure_timesteps - (shortcut_duration / 2) > 0).all()
        # departure_timesteps must be divisible by shortcut_duration
        assert (departure_timesteps % shortcut_duration == 0).all()


def test_flow_matching_batch_mask():
    # used in the training loop
    flow_matching_ratio = 0.75
    flow_match_mask = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.8]) <= flow_matching_ratio

    tensor = torch.randn_like(flow_match_mask.float())

    assert len(tensor[flow_match_mask]) == len([0.0, 0.25, 0.5, 0.75])
    assert len(tensor[~flow_match_mask]) == len([0.8])
