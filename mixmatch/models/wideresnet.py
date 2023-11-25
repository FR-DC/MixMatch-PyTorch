import math
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.nn.parallel
from torch.nn.functional import one_hot
from torchmetrics.functional import accuracy

from utils import SemiLoss, WeightEMA
import utils.interleave


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        stride,
        drop_rate=0.0,
        activate_before_residual=False,
    ):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(
            in_dim,
            out_dim,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_dim, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(
            out_dim,
            out_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.drop_rate = drop_rate
        self.equal_in_out = in_dim == out_dim
        if not self.equal_in_out:
            self.conv_shortcut = nn.Conv2d(
                in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False
            )
        else:
            self.conv_shortcut = None

        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if self.equal_in_out or not self.activate_before_residual:
            out = self.relu1(self.bn1(x))
        else:
            x = self.relu1(self.bn1(x))

        if self.equal_in_out:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))

        out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        if self.equal_in_out:
            y = torch.add(x, out)
        else:
            y = torch.add(self.conv_shortcut(x), out)

        return y


class NetworkBlock(nn.Module):
    def __init__(
        self,
        n_blocks,
        in_dim,
        out_dim,
        block,
        stride,
        drop_rate=0.0,
        activate_before_residual=False,
    ):
        super(NetworkBlock, self).__init__()
        layers = []
        for i in range(int(n_blocks)):
            layers.append(
                block(
                    i == 0 and in_dim or out_dim,
                    out_dim,
                    i == 0 and stride or 1,
                    drop_rate,
                    activate_before_residual,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        n_classes: int,
        depth: int = 28,
        width: int = 2,
        drop_rate: float = 0.0,
        seed: int = 42,
    ):
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        super().__init__()
        torch.manual_seed(seed)
        n_channels = [
            16,
            16 * width,
            32 * width,
            64 * width,
        ]
        blocks = (depth - 4) / 6
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            blocks,
            n_channels[0],
            n_channels[1],
            BasicBlock,
            1,
            drop_rate,
            activate_before_residual=True,
        )
        # 2nd block
        self.block2 = NetworkBlock(
            blocks, n_channels[1], n_channels[2], BasicBlock, 2, drop_rate
        )
        # 3rd block
        self.block3 = NetworkBlock(
            blocks, n_channels[2], n_channels[3], BasicBlock, 2, drop_rate
        )
        # global average pooling and classifier
        self.bn = nn.BatchNorm2d(n_channels[3], momentum=0.001)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(n_channels[3], n_classes)
        self.out_dim = n_channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                blocks = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / blocks))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.leaky_relu(self.bn(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.out_dim)
        return self.fc(x)


import pytorch_lightning as pl


# The eq=False is to prevent overriding hash
@dataclass(eq=False)
class WideResNetModule(pl.LightningModule):
    n_classes: int
    depth: int = 28
    width: int = 2
    drop_rate: float = 0.0
    seed: int = 42
    sharpen_temp: float = 0.5
    mix_beta_alpha: float = 0.75
    unl_loss_scale: float = 75
    ema_lr: float = 0.001
    lr: float = 0.002
    weight_decay: float = 0.0005

    # See our wiki for details on interleave
    interleave: bool = False

    train_loss_fn: SemiLoss = SemiLoss()

    def __post_init__(self):
        super().__init__()
        self.model = WideResNet(
            n_classes=self.n_classes,
            depth=self.depth,
            width=self.width,
            drop_rate=self.drop_rate,
            seed=self.seed,
        )
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.detach_()

        self.ema_updater = WeightEMA(
            model=self.model,
            ema_model=self.ema_model,
            ema_lr=self.ema_lr
        )

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
        y_unls: list[torch.Tensor] = [torch.softmax(self.ema_model(u), dim=1) for u in x_unls]
        # The sum will sum the tensors in the list, it doesn't reduce the tensors
        y_unl = sum(y_unls) / len(y_unls)
        return y_unl

    def training_step(self, batch, batch_idx):
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
            x_mix = utils.interleave(x_mix, batch_size)

            y_mix_pred = [self(x) for x in x_mix]

            # Un-interleave to shuffle back to original order
            y_mix_pred = utils.interleave(y_mix_pred, batch_size)

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

        loss_lbl, loss_unl = self.train_loss_fn(
            x_lbl=y_mix_lbl_pred,
            y_lbl=y_mix_lbl,
            x_unl=y_mix_unl_pred,
            y_unl=y_mix_unl,
        )
        loss_unl_scale = (
            (self.current_epoch + batch_idx / self.trainer.num_training_batches)
            / self.trainer.max_epochs
            * self.unl_loss_scale
        )
        loss = loss_lbl + loss_unl * loss_unl_scale

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_loss_lbl", loss_lbl)
        self.log("train_loss_unl", loss_unl)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.ema_model(x)
        loss = F.cross_entropy(y_pred, y.long())

        acc = accuracy(
            y_pred,
            y,
            task="multiclass",
            num_classes=y_pred.shape[1],
        )
        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    # PyTorch Lightning doesn't automatically no_grads the EMA step.
    # It's important to keep this to avoid a memory leak.
    @torch.no_grad()
    def on_after_backward(self) -> None:
        self.ema_updater.step()

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
