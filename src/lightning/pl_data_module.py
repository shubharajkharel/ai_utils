import pickle
import os
from typing import Literal, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


class PlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Dataset,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 10,
        pin_memory: bool = True,
        shuffle: bool = True,
        persistent_workers: bool = True,
        stratify: Optional[bool] = False,
        stratification_labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        drop_last: bool = True,
        **data_loader_kwargs: dict,  # useful for stratified sampling
    ):
        super().__init__()
        self.dataset = dataset
        self.random_seed = random_seed
        # this is required from batch_size tuner
        # https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.stratify = stratify
        self.stratification_labels = stratification_labels
        self.use_cache = use_cache

        self.data_loader_kwargs = {
            **data_loader_kwargs,
            "batch_size": batch_size,
            "num_workers": num_workers,  # number of subprocesses
            "pin_memory": pin_memory,  # faster transfer to GPU
            "shuffle": shuffle,  # reshuffle at every epoch
            # "persistent_workers": persistent_workers,  # keep workers alive
            "drop_last": True,  # drop last batch if smaller than batch_size
        }

        if stratify and stratification_labels is None:
            if hasattr(self.dataset, "labels"):
                self.stratification_labels = self.dataset.labels
            else:
                raise ValueError(
                    "When stratification is needed either: \n \
                        1. Pass stratification_labels as a tensor of labels \n \
                        2. Dataset must have a labels attribute"
                )

        self.train_idx, self.val_idx, self.test_idx = (
            self._create_stratified_idx() if self.stratify else self._create_idx()
        )

        self.train_data = Subset(self.dataset, self.train_idx)
        self.val_data = Subset(self.dataset, self.val_idx)
        self.predict_data = Subset(self.dataset, self.test_idx)
        self.test_data = Subset(self.dataset, self.test_idx)

    def setup(self, stage: Union[str, None] = None):
        pass
        # if stage == "fit":
        #     self.train_data = Subset(self.dataset, self.train_idx)
        #     self.val_data = Subset(self.dataset, self.val_idx)
        # elif stage == "test":
        #     self.test_data = Subset(self.dataset, self.test_idx)
        # elif stage == "predict":
        #     self.predict_data = Subset(self.dataset, self.test_idx)
        # elif stage == "validate":
        #     self.val_data = Subset(self.dataset, self.val_idx)

    def train_dataloader(self):
        return DataLoader(self.train_data, **self.data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(
            self.val_data, **{**self.data_loader_kwargs, "shuffle": False}
        )

    def test_dataloader(self):
        # return DataLoader(self.test_data, **self.data_loader_kwargs)
        return DataLoader(
            self.test_data, **{**self.data_loader_kwargs, "shuffle": False}
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_data, **{**self.data_loader_kwargs, "shuffle": False}
        )

    def _create_idx(self):
        # stage option is not used here
        # but is passed in by tuner
        train_len = int(self.train_size * len(self.dataset))
        val_len = int(self.val_size * len(self.dataset))
        test_len = len(self.dataset) - train_len - val_len
        assert test_len >= 0, ValueError("train_size + val_size must be less than 1")
        idx = np.arange(len(self.dataset))
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            np.random.shuffle(idx)
        train_idx = idx[:train_len]
        val_idx = idx[train_len : train_len + val_len]
        test_idx = idx[train_len + val_len :]

        return train_idx, val_idx, test_idx

    def _create_stratified_idx(self):
        # TODO: very slow currently, using pickle for cache
        cache_file = "stratified_idx.pkl"
        if self.use_cache:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    return pickle.load(f)

        if self.stratification_labels is None:
            raise ValueError(
                "stratification_labels must be provided when stratify is True"
            )
        train_idx, temp_idx = next(
            StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.train_size,
                random_state=self.random_seed,
            ).split(self.dataset, self.stratification_labels)
        )
        temp_labels = self.stratification_labels[temp_idx]
        val_idx, test_idx = next(
            StratifiedShuffleSplit(
                n_splits=1,
                train_size=self.val_size / (1 - self.train_size),
                random_state=self.random_seed,
            ).split(
                np.zeros(len(temp_idx)), temp_labels
            )  # zeros are dummy
        )

        # Cache the output
        indices = train_idx, val_idx, test_idx
        with open(cache_file, "wb") as f:
            pickle.dump(indices, f)

        return train_idx, val_idx, test_idx
