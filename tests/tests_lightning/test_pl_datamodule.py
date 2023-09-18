import unittest

import torch
from torch.utils.data import Dataset

from src.lightning.pl_data_module import PlDataModule


class SimpleDataset(Dataset):
    def __init__(self, length=1000):
        self.data = torch.randn(length, 10)
        self.targets = torch.randint(0, 2, (length,))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # classifciation
        cls.dataset = SimpleDataset()
        cls.data_module = PlDataModule(cls.dataset)
        cls.data_module.setup()

    def test_dtype(self):
        self.assertEqual(type(self.dataset[0][0]), torch.Tensor)
        self.assertEqual(self.dataset[0][1].dtype, torch.long)

    def test_len(self):
        train_len = len(self.data_module.train_data)
        val_len = len(self.data_module.val_data)
        test_len = len(self.data_module.test_data)
        self.assertEqual(train_len + val_len + test_len, len(self.dataset))

    def test_batch_size(self):
        data, _ = next(iter(self.data_module.train_dataloader()))
        self.assertEqual(len(data), self.data_module.batch_size)

        data, _ = next(iter(self.data_module.val_dataloader()))
        self.assertEqual(len(data), self.data_module.batch_size)

        data, _ = next(iter(self.data_module.test_dataloader()))
        self.assertEqual(len(data), self.data_module.batch_size)


if __name__ == "__main__":
    unittest.main()
