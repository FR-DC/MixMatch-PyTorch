from __future__ import annotations

from dataclasses import dataclass, KW_ONLY
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomCrop,
)

tf_preproc = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

tf_aug = transforms.Compose(
    [
        lambda x: torch.nn.functional.pad(
            x,
            (4,) * 4,
            mode="reflect",
        ),
        RandomCrop(32),
        RandomHorizontalFlip(),
    ]
)


@dataclass
class CIFAR10Subset(CIFAR10):
    _: KW_ONLY
    root: str
    idxs: Sequence[int] | None = None
    train: bool = True
    transform: Callable | None = None
    target_transform: Callable | None = None
    download: bool = False

    def __post_init__(self):
        super().__init__(
            root=self.root,
            train=self.train,
            transform=self.transform,
            target_transform=self.target_transform,
        )
        if self.idxs is not None:
            self.data = self.data[self.idxs]
            self.targets = np.array(self.targets)[self.idxs].tolist()


@dataclass
class CIFAR10SubsetKAug(CIFAR10Subset):
    _: KW_ONLY
    k_augs: int = 1
    aug: Callable = lambda x: x

    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return tuple(self.aug(img) for _ in range(self.k_augs)), target


def get_dataloaders(
    dataset_dir: Path | str,
    n_train_lbl: float = 0.005,
    n_train_unl: float = 0.980,
    batch_size: int = 48,
    num_workers: int = 0,
    seed: int | None = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """Get the dataloaders for the CIFAR10 dataset.

    Notes:
        The train_lbl_size and train_unl_size must sum to less than 1.
        The leftover data is used for the validation set.

    Args:
        dataset_dir: The directory where the dataset is stored.
        n_train_lbl: The size of the labelled training set.
        n_train_unl: The size of the unlabelled training set.
        batch_size: The batch size.
        num_workers: The number of workers for the dataloaders.
        seed: The seed for the random number generators. If None, then it'll be
            non-deterministic.

    Returns:
        4 DataLoaders: train_lbl_dl, train_unl_dl, val_unl_dl, test_dl
    """
    deterministic = seed is not None

    if deterministic:
        torch.manual_seed(seed)
        np.random.seed(seed)

    src_train_ds = CIFAR10(
        dataset_dir,
        train=True,
        download=True,
        transform=tf_preproc,
    )
    src_test_ds = CIFAR10(
        dataset_dir,
        train=False,
        download=True,
        transform=tf_preproc,
    )

    n_train = len(src_train_ds)
    n_train_unl = int(n_train * n_train_unl)
    n_train_lbl = int(n_train * n_train_lbl)
    n_val = int(n_train - n_train_unl - n_train_lbl)

    targets = np.array(src_train_ds.targets)
    ixs = np.arange(len(targets))
    ixs_train_unl, ixs_lbl = train_test_split(
        ixs,
        train_size=n_train_unl,
        stratify=targets,
    )
    lbl_targets = targets[ixs_lbl]

    ixs_val, ixs_train_lbl = train_test_split(
        ixs_lbl,
        train_size=n_val,
        stratify=lbl_targets,
    )

    ds_args = dict(root=dataset_dir, train=True, download=True, transform=tf_preproc)

    train_lbl_ds = CIFAR10SubsetKAug(
        **ds_args, idxs=ixs_train_lbl, k_augs=1, aug=tf_aug
    )
    train_unl_ds = CIFAR10SubsetKAug(
        **ds_args, idxs=ixs_train_unl, k_augs=2, aug=tf_aug
    )
    val_ds = CIFAR10Subset(**ds_args, idxs=ixs_val)

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_lbl_dl = DataLoader(train_lbl_ds, shuffle=True, drop_last=True, **dl_args)
    train_unl_dl = DataLoader(train_unl_ds, shuffle=True, drop_last=True, **dl_args)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)
    test_dl = DataLoader(src_test_ds, shuffle=False, **dl_args)

    return (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        src_train_ds.classes,
    )
