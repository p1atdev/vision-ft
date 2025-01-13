import torch

from src.modules.loss.shortcut import (
    prepare_self_consistency_targets,
    prepare_random_shortcut_distances,
)


def test_random_shortcut_distance():
    batch_size = 4

    inference_steps, shortcut_exponent, shortcut_distance_size, departure_timesteps = (
        prepare_random_shortcut_distances(
            batch_size=batch_size,
            max_pow=7,
        )
    )

    assert inference_steps.shape == (batch_size,)
    assert shortcut_exponent.shape == (batch_size,)
    assert shortcut_distance_size.shape == (batch_size,)
    assert departure_timesteps.shape == (batch_size,)


def test_flow_matching_batch_mask():
    # used in the training loop
    flow_matching_ratio = 0.75
    flow_match_mask = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.8]) <= flow_matching_ratio

    tensor = torch.randn_like(flow_match_mask.float())

    assert len(tensor[flow_match_mask]) == len([0.0, 0.25, 0.5, 0.75])
    assert len(tensor[~flow_match_mask]) == len([0.8])
