from src.torch.balance_data import BalancedDataset
import unittest

import torch
from torch.utils.data import Dataset


class UnbalancedDataset(Dataset):
    def __init__(self, length):
        self.length = length
        self.data = []
        for i in range(length):
            data = torch.randn(3)
            if i % 3 == 0:  # Change this ratio to make the dataset unbalanced
                target = 0
            else:
                target = 1
            self.data.append((data, target))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data, target = self.data[idx]
        return data, target


class TestBalancedDataset(unittest.TestCase):
    def test_balance(self):
        dummy_dataset = UnbalancedDataset(100)

        # Helper functions
        count_class1 = lambda x: len([item for item in x if item[1] == 1])
        count_class0 = lambda x: len([item for item in x if item[1] == 0])
        counts = lambda x: (count_class0(x), count_class1(x))

        with self.subTest("Test balanced dataset: None"):
            dataset = BalancedDataset(dummy_dataset)
            extracted_data = [dataset[i] for i in range(len(dataset))]
            self.assertEqual(*counts(extracted_data))

        with self.subTest("Test balanced dataset: Float"):
            dataset = BalancedDataset(dummy_dataset, sample_size=0.5)
            self.assertEqual(*counts(dataset))

        with self.subTest("Test balanced dataset: Adjusting for minority class"):
            with self.assertWarns(UserWarning):
                dataset = BalancedDataset(dummy_dataset, sample_size=999999)
                self.assertEqual(*counts(dataset))


if __name__ == "__main__":
    unittest.main()
