import unittest
import torch
from torch.utils.data import DataLoader
from src.torch.sample_torch_dataset import SampleTorchDataset
from src.torch.torch_sample_io import PyTorchSampleIO


class SimpleTorchModel(torch.nn.Module):
    def __init__(self):
        super(SimpleTorchModel, self).__init__()
        self.fc = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.fc(x)


class TestSampleTorchDataset(unittest.TestCase):
    def setUp(self):
        self.model = SimpleTorchModel()
        self.model.input_shape = [1, 28, 28]  # Update as per your model
        self.sample_io = PyTorchSampleIO

    def test_init(self):
        dataset = SampleTorchDataset(
            self.model, io_generator=self.sample_io, batch_size=10
        )
        self.assertEqual(len(dataset), 10)

    def test_get_item(self):
        dataset = SampleTorchDataset(
            self.model, io_generator=self.sample_io, batch_size=10
        )
        data, target = dataset.__getitem__(0)
        self.assertIsInstance(data, torch.Tensor)
        self.assertIsInstance(target, torch.Tensor)

    def test_inference(self):
        dataset = SampleTorchDataset(
            self.model, io_generator=self.sample_io, batch_size=10
        )
        dataloader = DataLoader(dataset, batch_size=2)
        for batch_idx, (data, target) in enumerate(dataloader):
            output = self.model(data)
            self.assertEqual(output.shape[0], 2)  # Matching the batch_size
            break  # Testing one batch is sufficient


if __name__ == "__main__":
    unittest.main()
