from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
from torch.nn.functional import one_hot
from torchmetrics.functional import accuracy

import mixmatch.utils.interleave
from mixmatch.utils.ema import WeightEMA


class LossUnlScale(Protocol):
    def __call__(self, progress: float) -> float:
        return progress * 75


# The eq=False is to prevent overriding hash
@dataclass(eq=False)
class MixMatchModule(pl.LightningModule):
    """PyTorch Lightning Module for MixMatch

    Notes:
        This performs MixMatch as described in the paper.
        https://arxiv.org/abs/1905.02249

        This module is designed to be used with any model, not only
        the WideResNet model.

        Furthermore, while it's possible to switch datasets, take a look
        at how we implement the CIFAR10DataModule's DataLoaders to see
        how to implement a new dataset.

    Args:
        model_fn: The function to use to create the model.
        n_classes: The number of classes in the dataset.
        sharpen_temp: The temperature to use for sharpening.
        mix_beta_alpha: The alpha to use for the beta distribution when mixing.
        loss_unl_scaler: The scale to use for the unsupervised loss.
        ema_lr: The learning rate to use for the EMA.
        lr: The learning rate to use for the optimizer.
        weight_decay: The weight decay to use for the optimizer.
    """

    model_fn: Callable[[], nn.Module]
    loss_unl_scaler: LossUnlScale
    n_classes: int = 10
    sharpen_temp: float = 0.5
    mix_beta_alpha: float = 0.75
    ema_lr: float = 0.001
    lr: float = 0.002
    weight_decay: float = 0.00004

    # See our wiki for details on interleave
    interleave: bool = False

    get_loss_lbl: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = F.cross_entropy
    # TODO: Not sure why this is different from MSELoss
    #  It's likely not a big deal, but it's worth investigating if we have
    #  too much time on our hands
    get_loss_unl: Callable[
        [torch.Tensor, torch.Tensor], torch.Tensor
    ] = lambda pred, tgt: torch.mean((torch.softmax(pred, dim=1) - tgt) ** 2)

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model_fn",
                "get_loss_lbl",
                "get_loss_unl",
                "loss_unl_scaler",
                "model",
            ]
        )
        self.model = self.model_fn()
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.ema_updater = WeightEMA(model=self.model, ema_model=self.ema_model)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def mix_up(
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mix up the data

        Args:
            x: The data to mix up.
            y: The labels to mix up.
            alpha: The alpha to use for the beta distribution.

        Returns:
            The mixed up data and labels.
        """
        ratio = np.random.beta(alpha, alpha)
        ratio = max(ratio, 1 - ratio)

        shuf_idx = torch.randperm(x.size(0))

        x_mix = ratio * x + (1 - ratio) * x[shuf_idx]
        y_mix = ratio * y + (1 - ratio) * y[shuf_idx]
        return x_mix, y_mix

    @staticmethod
    def sharpen(y: torch.Tensor, temp: float) -> torch.Tensor:
        """Sharpen the predictions by raising them to the power of 1 / temp

        Args:
            y: The predictions to sharpen.
            temp: The temperature to use.

        Returns:
            The probability-normalized sharpened predictions
        """
        y_sharp = y ** (1 / temp)
        # Sharpening will change the sum of the predictions.
        y_sharp /= y_sharp.sum(dim=1, keepdim=True)
        return y_sharp

    def guess_labels(
        self,
        x_unls: list[torch.Tensor],
    ) -> torch.Tensor:
        """Guess labels from the unlabelled data"""
        y_unls: list[torch.Tensor] = [
            torch.softmax(self.ema_model(u), dim=1) for u in x_unls
        ]
        # The sum will sum the tensors in the list, it doesn't reduce the tensors
        y_unl = sum(y_unls) / len(y_unls)
        return y_unl

    @property
    def progress(self):
        # Progress is a linear ramp from 0 to 1 over the course of training.
        return (
            self.global_step / self.trainer.num_training_batches
        ) / self.trainer.max_epochs

    def training_step(self, batch, batch_idx):
        # Progress is a linear ramp from 0 to 1 over the course of training.q
        (x_lbl, y_lbl), (x_unls, _) = batch
        x_lbl = x_lbl[0]
        y_lbl = one_hot(y_lbl.long(), num_classes=self.n_classes)

        with torch.no_grad():
            y_unl = self.guess_labels(x_unls=x_unls)
            y_unl = self.sharpen(y_unl, self.sharpen_temp)

        x = torch.cat([x_lbl, *x_unls], dim=0)
        y = torch.cat([y_lbl, y_unl, y_unl], dim=0)
        x_mix, y_mix = self.mix_up(x, y, self.mix_beta_alpha)

        if self.interleave:
            # This performs interleaving, see our wiki for details.
            batch_size = x_lbl.shape[0]
            x_mix = list(torch.split(x_mix, batch_size))

            # Interleave to get a consistent Batch Norm Calculation
            x_mix = mixmatch.utils.interleave.interleave(x_mix, batch_size)

            y_mix_pred = [self(x) for x in x_mix]

            # Un-interleave to shuffle back to original order
            y_mix_pred = mixmatch.utils.interleave.interleave(y_mix_pred, batch_size)

            y_mix_lbl_pred = y_mix_pred[0]
            y_mix_lbl = y_mix[:batch_size]
            y_mix_unl_pred = torch.cat(y_mix_pred[1:], dim=0)
            y_mix_unl = y_mix[batch_size:]
        else:
            batch_size = x_lbl.shape[0]
            y_mix_pred = self(x_mix)
            y_mix_lbl_pred = y_mix_pred[:batch_size]
            y_mix_unl_pred = y_mix_pred[batch_size:]
            y_mix_lbl = y_mix[:batch_size]
            y_mix_unl = y_mix[batch_size:]

        loss_lbl = self.get_loss_lbl(y_mix_lbl_pred, y_mix_lbl)
        loss_unl = self.get_loss_unl(y_mix_unl_pred, y_mix_unl)

        loss_unl_scale = self.loss_unl_scaler(progress=self.progress)

        loss = loss_lbl + loss_unl * loss_unl_scale

        self.log("loss_unl_scale", loss_unl_scale, prog_bar=True)
        self.log("train_loss", loss)
        self.log("train_loss_lbl", loss_lbl)
        self.log("train_loss_unl", loss_unl)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.ema_model(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(y_pred, y, task="multiclass", num_classes=y_pred.shape[1])
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # PyTorch Lightning doesn't automatically no_grads the EMA step.
    # It's important to keep this to avoid a memory leak.
    @torch.no_grad()
    def on_after_backward(self) -> None:
        self.ema_updater.update(self.ema_lr)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
