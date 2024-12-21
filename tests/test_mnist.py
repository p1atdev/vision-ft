import torch

from src.dataset.mnist import MnistDatasetConfig
from src.models.mnist import MnistModelForTraining, MnistConfig
from src.config import TrainConfig


from accelerate import Accelerator


def test_train_step():
    # Initialize model configuration and model for training
    config = MnistConfig()
    model_for_training = MnistModelForTraining(
        accelerator=Accelerator(cpu=True),
        config=TrainConfig(
            **{
                "model": config,
                "dataset": MnistDatasetConfig(batch_size=2),
            }
        ),
    )
    model_for_training.setup_model()
    model_for_training.setup_optimizer()
    model_for_training.sanity_check()

    # Create a mock batch with pixel values and targets
    pixel_values = torch.randn(32, config.num_pixels)  # batch size of 32
    targets = torch.randint(0, config.num_labels, (32,))  # batch size of 32

    batch = (pixel_values, targets)

    # Call the train_step method
    loss = model_for_training.train_step(batch)

    # Assert that the returned loss is a tensor and has the expected shape
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == torch.Size([])  # loss should be a scalar tensor
