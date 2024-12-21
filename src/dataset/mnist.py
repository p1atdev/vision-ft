from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as F

from datasets import load_dataset, Dataset, DatasetDict

from .util import DatasetConfig


class MnistDatasetConfig(DatasetConfig):
    repo_id: str = "ylecun/mnist"

    def get_dataset(self):
        ds = load_dataset(self.repo_id)
        assert isinstance(ds, DatasetDict)

        return MnistDataset(ds["train"]), MnistDataset(ds["test"])


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    tensor = F.pil_to_tensor(image)
    tensor = tensor / 255.0
    return tensor


class MnistDataset(data.Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx]["image"]  # PngImageFile
        label = self.dataset[idx]["label"]

        # pil to tensor
        image_tensor = pil_to_tensor(image)
        label_tensor = torch.LongTensor([label])

        assert image_tensor.shape == torch.Size([1, 28, 28])
        assert label_tensor.shape == torch.Size([1])

        return image_tensor, label_tensor
