from typing import Optional
from torch.utils.data import random_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset

import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader


class PlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 42,
        train_size: float = 0.7,
        val_size: float = 0.15,
        batch_size: int = 128,
    ):
        super().__init__()
        self.dataset = dataset
        self.seed = seed

        # this is required from batch_size tuner
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
        self.batch_size = batch_size

        self.train_size = train_size
        self.val_size = val_size

    def setup(self, stage: Optional[str] = None):
        # stage option is not used here
        # but is passed in by tuner
        train_len = int(self.train_size * len(self.dataset))
        val_len = int(self.val_size * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        assert test_len >= 0, ValueError("train_size + val_size must be less than 1")
        self.train_data, self.val_data, self.test_data = random_split(
            self.dataset,
            [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(self.seed),
        )
        self.predict_data = self.test_data

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size)
