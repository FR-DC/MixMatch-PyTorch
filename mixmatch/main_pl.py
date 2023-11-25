import pytorch_lightning as pl
import torch

from mixmatch.dataset.cifar10 import SSLCIFAR10DataModule
from mixmatch.models.wideresnet import WideResNet
from mixmatch.models.mixmatch_module import MixMatchModule

epochs: int = 100
batch_size: int = 64
k_augs: int = 2
lr: float = 0.002
weight_decay: float = 0.00004
ema_lr: float = 0.001
train_iters: int = 1024
unl_loss_scale: float = 75
mix_beta_alpha: float = 0.75
sharpen_temp: float = 0.5
device: str = "cuda"
seed: int | None = 42
train_lbl_size: float = 0.005
train_unl_size: float = 0.980

dm = SSLCIFAR10DataModule(
    dir="../tests/data",
    train_lbl_size=train_lbl_size,
    train_unl_size=train_unl_size,
    batch_size=batch_size,
    train_iters=train_iters,
    seed=seed,
    k_augs=k_augs,
    num_workers=32,
)

mm_model = MixMatchModule(
    model_fn=lambda: WideResNet(
        n_classes=10, depth=28, width=2, drop_rate=0.0, seed=seed
    ),
    n_classes=10,
    sharpen_temp=sharpen_temp,
    mix_beta_alpha=mix_beta_alpha,
    unl_loss_scale=unl_loss_scale,
    ema_lr=ema_lr,
    lr=lr,
    weight_decay=weight_decay,
)

torch.set_float32_matmul_precision("high")

trainer = pl.Trainer(max_epochs=epochs, accelerator="gpu")

trainer.fit(mm_model, dm)
