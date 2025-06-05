import torch
from PIL import Image
import random

from transformers import set_seed

from modules.reward.pickscore import PickScoreRewardModel


def test_pick_score_reward_model():
    set_seed(42)  # For reproducibility

    # Initialize the model
    device = torch.device("cpu")  # Use "cuda" if you have a GPU
    reward_model = PickScoreRewardModel(device=device)

    # Create dummy images and prompts
    images = [
        Image.new("RGB", (512, 512), color=(255, 0, 0)),
    ] + [
        Image.new(
            "RGB",
            (512, 512),
            color=(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255),
            ),
        )
    ] * 4  # 1 red image and 4 random images
    prompts = ["A solid red"] * 5

    # Call the model
    scores = reward_model(images, prompts)

    # Check the output shape
    assert scores.shape == (5,), "Output should be a tensor of shape (5,)"

    # the first image should have the highest score
    assert scores[0] > scores[1:].max(), "The first image should have the highest score"
