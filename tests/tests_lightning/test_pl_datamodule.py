import unittest
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset

from src.lightning.pl_data_module import PlDataModule


class SimpleDataset(Dataset):
    def __init__(self, length=10000):
        self.data = torch.randn(length, 10)
        self.targets = torch.randint(0, 2, (length,))
        self.labels = self.targets

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
        cls.data_module.setup(stage="fit")
        cls.data_module.setup(stage="test")

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

    def test_stratification(self):
        stratified_data_module = PlDataModule(
            self.dataset,
            stratify=True,
            stratification_labels=self.dataset.targets,
        )
        stratified_data_module.setup(stage="fit")
        train_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.train_data)]
        )
        val_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.val_data)]
        )
        stratified_data_module.setup(stage="test")
        test_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.test_data)]
        )

        total_targets = len(self.dataset.targets)

        train_ratio = train_targets.sum().item() / len(train_targets)
        val_ratio = val_targets.sum().item() / len(val_targets)
        test_ratio = test_targets.sum().item() / len(test_targets)
        original_ratio = self.dataset.targets.sum().item() / total_targets

        # Assuming a tolerance of 2%
        threshold = 0.02
        self.assertAlmostEqual(train_ratio, original_ratio, delta=threshold)
        self.assertAlmostEqual(val_ratio, original_ratio, delta=threshold)
        self.assertAlmostEqual(test_ratio, original_ratio, delta=threshold)


if __name__ == "__main__":
    unittest.main()
