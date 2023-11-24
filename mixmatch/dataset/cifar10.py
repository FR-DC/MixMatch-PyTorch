from __future__ import annotations

from dataclasses import dataclass, KW_ONLY, field
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


import pytorch_lightning as pl


@dataclass
class CIFAR10DataModule(pl.LightningDataModule):
    dir: Path | str
    n_train_lbl: float = 0.005
    n_train_unl: float = 0.980
    batch_size: int = 48
    num_workers: int = 0
    seed: int | None = 42
    train_lbl_ds: CIFAR10Subset = field(init=False)
    train_unl_ds: CIFAR10Subset = field(init=False)
    val_ds: CIFAR10Subset = field(init=False)
    test_ds: CIFAR10 = field(init=False)

    def __post_init__(self):
        super().__init__()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.ds_args = dict(
            root=self.dir, train=True, download=True, transform=tf_preproc
        )
        self.dl_args = dict(batch_size=self.batch_size, num_workers=self.num_workers)

    def setup(self, stage: str | None = None):
        src_train_ds = CIFAR10(
            self.dir,
            train=True,
            download=True,
            transform=tf_preproc,
        )
        self.test_ds = CIFAR10(
            self.dir,
            train=False,
            download=True,
            transform=tf_preproc,
        )

        n_train = len(src_train_ds)
        n_train_unl = int(n_train * self.n_train_unl)
        n_train_lbl = int(n_train * self.n_train_lbl)
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
        self.train_lbl_ds = CIFAR10SubsetKAug(
            **self.ds_args, idxs=ixs_train_lbl, k_augs=1, aug=tf_aug
        )
        self.train_unl_ds = CIFAR10SubsetKAug(
            **self.ds_args, idxs=ixs_train_unl, k_augs=2, aug=tf_aug
        )
        self.val_ds = CIFAR10Subset(**self.ds_args, idxs=ixs_val)

    def train_lbl_dataloader(self):
        return DataLoader(
            self.train_lbl_ds, shuffle=True, drop_last=True, **self.dl_args
        )

    def train_unl_dataloader(self):
        return DataLoader(
            self.train_unl_ds, shuffle=True, drop_last=True, **self.dl_args
        )

    def val_dataloader(self):
        return DataLoader(self.val_ds, shuffle=False, **self.dl_args)

    def test_dataloader(self):
        return DataLoader(self.test_ds, shuffle=False, **self.dl_args)
