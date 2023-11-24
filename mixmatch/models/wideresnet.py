import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            self.conv_shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, padding=0, bias=False)
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

class WideResNetModule(pl.LightningModule):
    def __init__(
        self,
        n_classes: int,
        depth: int = 28,
        width: int = 2,
        drop_rate: float = 0.0,
        seed: int = 42,
    ):
        super().__init__()
        self.model = WideResNet(
            n_classes=n_classes,
            depth=depth,
            width=width,
            drop_rate=drop_rate,
            seed=seed,
        )

    def forward(self, x):
        return self.model(x)

    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.model(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log("train_loss", loss)
    #     return loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.model(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log("val_loss", loss)
    #
    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.model.parameters(), lr=0.002)