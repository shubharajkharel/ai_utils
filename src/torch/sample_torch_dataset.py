import torch
from torch.utils.data import Dataset

from .simple_torch_models import SimpleTorchCNNModel
from ..lightning.pl_data_module import PlDataModule
from .torch_sample_io import PyTorchSampleIO


class SampleTorchDataset(Dataset):
    def __init__(
        self,
        model: torch.nn.Module,
        io_generator=PyTorchSampleIO,
        transform=None,
        batch_size=100,
        device=None,
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
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = torch.stack(self.data).to(device)
        self.targets = torch.stack(self.targets).to(device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx]) if self.transform else self.data[idx]
        target = self.targets[idx]
        return data, target


if __name__ == "__main__":
    # Initialize the dataset
    model_torch = SimpleTorchCNNModel()
    dataset = SampleTorchDataset(model_torch)
    print(f"Dataset length: {len(dataset)}")
    print(f"Dataset shape: {dataset.data.shape}")

    datamodule = PlDataModule(dataset)
