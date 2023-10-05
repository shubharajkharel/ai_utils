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
    # @classmethod
    # def setUpClass(cls):
    #     # classifciation
    #     cls.dataset = SimpleDataset()
    #     cls.data_module = PlDataModule(dataset=cls.dataset)

    def test_dtype(self):
        data_module = PlDataModule(dataset=SimpleDataset())
        self.assertEqual(type(data_module.dataset[0][0]), torch.Tensor)
        self.assertEqual(data_module.dataset[0][1].dtype, torch.long)

    def test_len(self):
        data_module = PlDataModule(dataset=SimpleDataset())
        train_len = len(data_module.train_dataset)
        val_len = len(data_module.val_dataset)
        test_len = len(data_module.test_dataset)
        self.assertEqual(train_len + val_len + test_len, len(data_module.dataset))

    def test_batch_size(self):
        data_module = PlDataModule(dataset=SimpleDataset(), batch_size=32)
        data, _ = next(iter(data_module.train_dataloader()))
        self.assertEqual(len(data), data_module.batch_size)

        data, _ = next(iter(data_module.val_dataloader()))
        self.assertEqual(len(data), data_module.batch_size)

        data, _ = next(iter(data_module.test_dataloader()))
        self.assertEqual(len(data), data_module.batch_size)

    def test_stratification(self):
        dataset = SimpleDataset()
        stratified_data_module = PlDataModule(
            dataset=dataset, stratify=True, stratification_labels=dataset.targets
        )
        train_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.train_dataset)]
        )
        val_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.val_dataset)]
        )
        stratified_data_module.setup(stage="test")
        test_targets = torch.cat(
            [y for _, y in DataLoader(stratified_data_module.test_dataset)]
        )

        total_targets = len(stratified_data_module.dataset.targets)

        train_ratio = train_targets.sum().item() / len(train_targets)
        val_ratio = val_targets.sum().item() / len(val_targets)
        test_ratio = test_targets.sum().item() / len(test_targets)
        original_ratio = dataset.targets.sum().item() / total_targets

        # Assuming a tolerance of 2%
        threshold = 0.05
        self.assertAlmostEqual(train_ratio, original_ratio, delta=threshold)
        self.assertAlmostEqual(val_ratio, original_ratio, delta=threshold)
        self.assertAlmostEqual(test_ratio, original_ratio, delta=threshold)

    def test_data_needs_paritioning(self):
        with self.subTest("none provided"):
            with self.assertRaises(ValueError):
                PlDataModule()

        with self.subTest("dataset provided"):
            data_module = PlDataModule(dataset=SimpleDataset())
            data_module.train_dataset = None
            data_module.val_dataset = None
            data_module.test_dataset = None
            self.assertEqual(data_module._dataset_needs_paritioning(), True)

        with self.subTest("dataset not provided"):
            data_module = PlDataModule(dataset=SimpleDataset())
            data_module.dataset = None
            data_module.train_dataset = SimpleDataset()
            data_module.val_dataset = SimpleDataset()
            data_module.test_dataset = SimpleDataset()
            self.assertEqual(data_module._dataset_needs_paritioning(), False)

        with self.subTest("both provided"):
            data_module = PlDataModule(SimpleDataset())
            data_module.dataset = SimpleDataset()
            data_module.train_dataset = SimpleDataset()
            data_module.val_dataset = SimpleDataset()
            data_module.test_dataset = SimpleDataset()
            with self.assertRaises(ValueError):
                data_module._dataset_needs_paritioning()


if __name__ == "__main__":
    unittest.main()
