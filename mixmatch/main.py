import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from mixmatch.dataset.cifar10 import get_dataloaders
from models.wideresnet import WideResNet
from utils.ema import WeightEMA
from utils.eval import validate, train
from utils.loss import SemiLoss


def main(
    *,
    epochs: int = 1024,
    batch_size: int = 64,
    lr: float = 0.002,
    train_iteration: int = 1024,
    ema_wgt_decay: float = 0.999,
    unl_loss_scale: float = 75,
    mix_beta_alpha: float = 0.75,
    sharpen_temp: float = 0.5,
    device: str = "cuda",
    seed: int | None = 42,
    train_lbl_size: int = 0.005,
    train_unl_size: int = 0.980,
):
    """The main function to run the MixMatch algorithm

    Args:
        epochs: Number of epochs to run.
        batch_size: The batch size to use.
        lr: The learning rate to use.
        train_iteration: The number of iterations to train for.
        ema_wgt_decay: The weight decay to use for the EMA model.
        unl_loss_scale: The scaling factor for the unlabeled loss.
        mix_beta_alpha: The beta alpha to use for the mixup.
        sharpen_temp: The temperature to use for sharpening.
        device: The device to use.
        seed: The seed to use. If None, then it'll be non-deterministic.
        train_lbl_size: The size of the labeled training set.
        train_unl_size: The size of the unlabeled training set.
    """
    deterministic = seed is not None

    if deterministic:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Data
    print(f"==> Preparing cifar10")

    (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        classes,
    ) = get_dataloaders(
        dataset_dir="./data",
        n_train_lbl=train_lbl_size,
        n_train_unl=train_unl_size,
        batch_size=batch_size,
        seed=seed,
    )

    # Model
    print("==> creating WRN-28-2")

    model = WideResNet(num_classes=len(classes)).to(device)
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.detach_()

    train_loss_fn = SemiLoss()
    val_loss_fn = nn.CrossEntropyLoss()
    train_optim = optim.Adam(model.parameters(), lr=lr)

    ema_optim = WeightEMA(model, ema_model, ema_wgt_decay=ema_wgt_decay, lr=lr)

    test_accs = []
    best_acc = 0
    # Train and val
    for epoch in range(epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, epochs, lr))

        train_loss, train_lbl_loss, train_unl_loss = train(
            train_lbl_dl=train_lbl_dl,
            train_unl_dl=train_unl_dl,
            model=model,
            optim=train_optim,
            ema_optim=ema_optim,
            loss_fn=train_loss_fn,
            epoch=epoch,
            device=device,
            train_iters=train_iteration,
            unl_loss_scale=unl_loss_scale,
            mix_beta_alpha=mix_beta_alpha,
            epochs=epochs,
            sharpen_temp=sharpen_temp,
        )

        def val_ema(dl: DataLoader):
            return validate(
                valloader=dl,
                model=ema_model,
                loss_fn=val_loss_fn,
                device=device,
            )

        _, train_acc = val_ema(train_lbl_dl)
        val_loss, val_acc = val_ema(val_dl)
        test_loss, test_acc = val_ema(test_dl)

        best_acc = max(val_acc, best_acc)
        test_accs.append(test_acc)

        print(
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | "
            f"Best Acc: {best_acc:.3f} | "
            f"Mean Acc: {np.mean(test_accs[-20:]):.3f} | "
            f"LR: {lr:.5f} | "
            f"Train Loss X: {train_lbl_loss:.3f} | "
            f"Train Loss U: {train_unl_loss:.3f} "
        )

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])
