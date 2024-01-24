import os
import pickle
from typing import Optional, Union

# import pytorch_lightning as pl
import lightning as pl
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset


class PlDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: Union[Dataset, None] = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        random_seed: int = 42,
        batch_size: int = 128,
        num_workers: int = 10,
        pin_memory: bool = True,
        shuffle: bool = False,
        persistent_workers: bool = True,
        stratify: Optional[bool] = False,
        stratification_labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        drop_last: bool = True,
        train_dataset: Union[torch.utils.data.Dataset, None] = None,
        test_dataset: Union[torch.utils.data.Dataset, None] = None,
        val_dataset: Union[torch.utils.data.Dataset, None] = None,
        # device: Union[ str, torch.device, None ] = None,  # only for mps, done auto for others
        **data_loader_kwargs: dict,  # useful for stratified sampling
    ):
        super().__init__()
        self.dataset = dataset
        self.random_seed = random_seed
        self.stratify = stratify
        self.stratification_labels = stratification_labels
        self.use_cache = use_cache
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size  # batch_size is required from batch_size tuner
        self.train_size = train_size
        self.val_size = val_size

        self.data_loader_kwargs = {
            **data_loader_kwargs,
            "batch_size": batch_size,
            "num_workers": num_workers,  # number of subprocesses
            "pin_memory": pin_memory,  # faster transfer to GPU
            "shuffle": shuffle,  # reshuffle at every epoch
            # "persistent_workers": persistent_workers,  # keep workers alive
            "drop_last": drop_last,  # drop last batch if smaller than batch_size
        }

        if (
            self._dataset_needs_paritioning()
        ):  # TODO: move this to functions acting on dataset
            self._create_test_train_val_data()  # updates train/test/val data

    def _create_test_train_val_data(self):
        if self.stratify and self.stratification_labels is None:
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
        self.train_dataset = Subset(self.dataset, self.train_idx)
        self.val_dataset = Subset(self.dataset, self.val_idx)
        self.predict_dataset = Subset(self.dataset, self.test_idx)
        self.test_dataset = Subset(self.dataset, self.test_idx)

    def _dataset_needs_paritioning(self):
        train_test_val_provided = all(
            [
                self.train_dataset is not None,
                self.val_dataset is not None,
                self.test_dataset is not None,
            ]
        )

        dataset_provided = self.dataset is not None

        if not dataset_provided and not train_test_val_provided:
            raise ValueError(
                "Either dataset or train_data, val_data, test_data must be provided"
            )

        if dataset_provided and train_test_val_provided:
            raise ValueError(
                "Only one of dataset or train/test/val data must be provided"
            )

        return True if dataset_provided else False

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
        return DataLoader(self.train_dataset, **self.data_loader_kwargs)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **{**self.data_loader_kwargs})

    def test_dataloader(self):
        # return DataLoader(self.test_data, **self.data_loader_kwargs)
        return DataLoader(self.test_dataset, **{**self.data_loader_kwargs})

    def predict_dataloader(self):
        # TODO: remove the proxy
        return DataLoader(self.test_dataset, **{**self.data_loader_kwargs})

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
