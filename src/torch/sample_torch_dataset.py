import torch
from torch.utils.data import Dataset

from src.misc.simple_models import SimpleTorchCNNModel
from src.pytorch_lightning.pl_data_module import PlDataModule
from src.torch.torch_sample_io import PyTorchSampleIO


class SampleTorchDataset(Dataset):
    def __init__(
        self,
        model: torch.nn.Module,
        io_generator=PyTorchSampleIO,
        transform=None,
        batch_size=100,
    ):
        self.model = model
        self.io_generator = io_generator
        self.transform = transform
        self.data = []
        self.targets = []
        for _ in range(batch_size):
            data, target = io_generator(model)
            self.data.append(data)
            self.targets.extend(target)
        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx]) if self.transform else self.data[idx]
        target = self.targets[idx]
        return data, target


if __name__ == "__main__":
    from utils.pytorch_lightning.pl_data_module import PlDataModule

    # Initialize the dataset
    model_torch = SimpleTorchCNNModel()
    dataset = SampleTorchDataset(model_torch)
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset shape: {dataset.data.shape}")

    datamodule = PlDataModule(dataset)
