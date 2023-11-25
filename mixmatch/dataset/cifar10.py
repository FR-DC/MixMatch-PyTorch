from __future__ import annotations

from dataclasses import dataclass, KW_ONLY, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler
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


# TODO: We should make this dataset agnostic, so we can use it for other
#       datasets.
@dataclass
class SSLCIFAR10DataModule(pl.LightningDataModule):
    """The CIFAR10 datamodule for semi-supervised learning.

    Notes:
        This datamodule is configured for SSL on CIFAR10.

        The major difference is that despite the labelled data being smaller
        than the unlabelled data, the dataloader will sample with replacement
        to match training iterations. Hence, each epoch will have the same
        number of training iterations for labelled and unlabelled data.

        The batch size, thus, doesn't affect the number of training iterations,
        each iteration will have the specified batch size.

        For example:
            train_lbl_size = 0.005 (250)
            train_unl_size = 0.980 (49000)
            batch_size = 48
            train_iters = 1024

            In pseudocode

            for epoch in range(epochs):
                for train_iter in range(1024):
                    lbl = sample(lbl_pool, 48)
                    unl = sample(unl_pool, 48)

            Each epoch will have 1024 training iterations.
            Each training iteration will pull 48 labelled and 48 unlabelled
            samples from the above pools, with replacement. Therefore, unlike
            traditional dataloaders, we can see repeated samples in the same
            epoch. (replacement=False in our RandomSampler only prevents
            replacements within a minibatch)

    Args:
        dir: The directory to store the data.
        train_lbl_size: The size of the labelled training set.
        train_unl_size: The size of the unlabelled training set.
        batch_size: The batch size to use.
        train_iters: The number of training iterations per epoch.
        seed: The seed to use for reproducibility. If None, no seed is used.
        k_augs: The number of augmentations to use for unlabelled data.
        num_workers: The number of workers to use for the dataloaders.
        persistent_workers: Whether to use persistent workers for the dataloaders.
        pin_memory: Whether to pin memory for the dataloaders.
    """
    dir: Path | str
    train_lbl_size: float = 0.005
    train_unl_size: float = 0.980
    batch_size: int = 48
    train_iters: int = 1024
    seed: int | None = 42
    k_augs: int = 2
    num_workers: int = 0
    persistent_workers: bool = True
    pin_memory: bool = True
    train_lbl_ds: CIFAR10 = field(init=False)
    train_unl_ds: CIFAR10 = field(init=False)
    val_ds: CIFAR10 = field(init=False)
    test_ds: CIFAR10 = field(init=False)

    def __post_init__(self):
        super().__init__()
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.ds_args = dict(
            root=self.dir, train=True, download=True, transform=tf_preproc
        )
        self.dl_args = dict(
            batch_size=self.batch_size,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
        )

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
        n_train_unl = int(n_train * self.train_unl_size)
        n_train_lbl = int(n_train * self.train_lbl_size)
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
            **self.ds_args, idxs=ixs_train_unl, k_augs=self.k_augs, aug=tf_aug
        )
        self.val_ds = CIFAR10Subset(**self.ds_args, idxs=ixs_val)

    def train_dataloader(self) -> list[DataLoader]:
        """The training dataloader returns a list of two dataloaders.

        Notes:
            This train dataloader is special in that
            1) The labelled and unlabelled are sampled separately.
            2) Despite labelled being smaller than unlabelled, the dataloader
               will sample with replacement to match training iterations.

            The num_samples supplied to the sampler is the exact number of
            samples, so we need to multiply by the batch size.

        Returns:
            A list of two dataloaders, the first for labelled data, the second
            for unlabelled data.
        """
        lbl_workers = self.num_workers // (self.k_augs + 1)
        unl_workers = self.num_workers - lbl_workers
        return [
            DataLoader(
                self.train_lbl_ds,
                sampler=RandomSampler(
                    self.train_lbl_ds,
                    num_samples=self.batch_size * self.train_iters,
                    replacement=False,
                ),
                num_workers=lbl_workers,
                **self.dl_args,
            ),
            DataLoader(
                self.train_unl_ds,
                sampler=RandomSampler(
                    self.train_unl_ds,
                    num_samples=self.batch_size * self.train_iters,
                    replacement=False,
                ),
                num_workers=unl_workers,
                **self.dl_args,
            ),
        ]

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds, shuffle=False, **self.dl_args, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_ds, shuffle=False, **self.dl_args, num_workers=self.num_workers
        )
