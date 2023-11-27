from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch

from mixmatch.dataset.cifar10 import SSLCIFAR10DataModule
from mixmatch.models.mixmatch_module import MixMatchModule
from mixmatch.models.wideresnet import WideResNet

epochs: int = 100
batch_size: int = 64
k_augs: int = 2
# Scale LR due to removed interleaving
lr: float = 0.002 * np.sqrt((k_augs + 1))
weight_decay: float = 0.00004
ema_lr: float = 0.005
train_iters: int = 1024
mix_beta_alpha: float = 0.75
sharpen_temp: float = 0.5
device: str = "cuda"
seed: int | None = 42
train_lbl_size: float = 0.005
train_unl_size: float = 0.980

# This function linearly increases scaler from 0 to MAX_UNL_SCALER over EPOCHS_TO_MAX_UNL_SCALER epochs, then
# keeps it at MAX_UNL_SCALER for the rest of the training.
#
#                ^
#                |
# MAX_UNL_SCALER |   +---------------------->
#                |  /.
#                | / .
#                |/  .
#                +---+---------------------->
#                0 EPOCHS_TO_MAX_UNL_SCALER EPOCHS

epochs_to_max_unl_scaler: int = 100
max_unl_scaler: int = 50
loss_unl_scaler = lambda progress: min(
    progress * epochs * max_unl_scaler / epochs_to_max_unl_scaler, max_unl_scaler
)

dm = SSLCIFAR10DataModule(
    dir=Path(__file__).parents[1] / "tests/data",
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
    loss_unl_scaler=loss_unl_scaler,
    ema_lr=ema_lr,
    lr=lr,
    weight_decay=weight_decay,
)

torch.set_float32_matmul_precision("high")

trainer = pl.Trainer(
    max_epochs=epochs,
    accelerator="gpu",
    callbacks=[
        pl.callbacks.LearningRateMonitor(),
        pl.callbacks.StochasticWeightAveraging(swa_lrs=lr),
        pl.callbacks.ModelCheckpoint(
            monitor="val_acc",
            filename="mm-{epoch:02d}-{val_acc:.2f}",
            save_top_k=1,
            mode="max",
        ),
    ],
)

trainer.fit(mm_model, dm)
