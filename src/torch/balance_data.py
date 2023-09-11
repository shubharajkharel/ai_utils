import logging
import torch
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset
from typing import Union, List
import warnings


class BalancedDataset(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        sample_size: Union[int, float, None] = None,
    ):
        # NOTE: this assumes that dataset getattr gives data, targets
        self.dataset = dataset
        self.targets = self.dataset[:][1].squeeze()
        self.classes = self.targets.unique()
        if len(self.classes) < 2:
            raise ValueError("Dataset has only one class")
        idx = torch.arange(len(dataset))
        # number of samples per class
        cls_count = self.find_cls_count(sample_size)
        # idx of samples with equal number of samples per class
        sample_idx = self.compute_sample_idx(
            idx, cls_count_list=[cls_count] * len(self.classes)
        )
        self.subset = torch.utils.data.Subset(dataset, sample_idx)

    def compute_sample_idx(
        self, idx: torch.Tensor, cls_count_list: List[int]
    ) -> torch.Tensor:
        idx_cls = []
        for cls, count in zip(self.classes, cls_count_list):
            idx_sub = idx[(self.targets == cls).squeeze()]
            idx_sub = idx_sub[:count]
            idx_cls.append(idx_sub)
        sample_idx = torch.cat(idx_cls, dim=0)
        # prevent ordered by class
        sample_idx = sample_idx[torch.randperm(len(sample_idx))]
        return sample_idx

    def find_cls_count(self, sample_length: Union[int, float, None]) -> int:
        minority_cls_count = self.targets.bincount().min()
        if isinstance(sample_length, float):
            # this assumes the subset dataset is not assigned yet
            count_per_cls = int(sample_length * len(self.dataset)) // len(self.classes)
        elif isinstance(sample_length, int):
            count_per_cls = sample_length // len(self.classes)
        elif isinstance(sample_length, type(None)):
            count_per_cls = minority_cls_count
        else:
            raise ValueError()

        if count_per_cls > minority_cls_count:
            warn_message = (
                f"Requested cls count {count_per_cls} > size of minority class {minority_cls_count},"
                f" using {minority_cls_count} instead"
            )
            # logging.warning(warn_message)
            warnings.warn(warn_message)  # this is used in unittest

            count_per_cls = minority_cls_count
        return count_per_cls

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        return self.subset[idx]


if __name__ == "__main__":
    from src.torch_dataset import WaveformDataset

    dataset = WaveformDataset(use_scenario="classification")
    print(f"Dataset length: {len(dataset)}")
    balanced_dataset = BalancedDataset(dataset, sample_size=0.001)
    print(f"Balanced Dataset length: {len(balanced_dataset)}")
