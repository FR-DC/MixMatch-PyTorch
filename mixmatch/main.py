import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader

import mixmatch.dataset.cifar10 as dataset
import mixmatch.models.wideresnet as models
from utils.ema import WeightEMA
from utils.eval import validate, train
from utils.loss import SemiLoss


def main(
    *,
    epochs: int = 1024,
    batch_size: int = 64,
    lr: float = 0.002,
    n_labeled: int = 250,
    train_iteration: int = 1024,
    ema_decay: float = 0.999,
    lambda_u: float = 75,
    alpha: float = 0.75,
    t: float = 0.5,
    device: str = "cuda",
    seed: int = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    best_acc = 0

    # Data
    print(f"==> Preparing cifar10")

    (
        labeled_trainloader,
        unlabeled_trainloader,
        val_loader,
        test_loader,
    ) = dataset.get_cifar10(
        "./data", n_labeled, batch_size=batch_size, seed=seed
    )

    # Model
    print("==> creating WRN-28-2")

    model = models.WideResNet(num_classes=10).to(device)
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.detach_()

    # cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay, lr=lr)

    test_accs = []
    # Train and val
    for epoch in range(epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, epochs, lr))

        train_loss, train_loss_x, train_loss_u = train(
            train_lbl_dl=labeled_trainloader,
            train_unl_dl=unlabeled_trainloader,
            model=model,
            optim=optimizer,
            ema_optim=ema_optimizer,
            loss_fn=train_criterion,
            epoch=epoch,
            device="cuda",
            train_iters=train_iteration,
            lambda_u=lambda_u,
            mix_beta_alpha=alpha,
            epochs=epochs,
            sharpen_temp=t,
        )

        def val_ema(dl: DataLoader):
            return validate(
                valloader=dl,
                model=ema_model,
                loss_fn=criterion,
                device=device,
            )

        _, train_acc = val_ema(labeled_trainloader)
        val_loss, val_acc = val_ema(val_loader)
        test_loss, test_acc = val_ema(test_loader)

        best_acc = max(val_acc, best_acc)
        test_accs.append(test_acc)

        print(
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | "
            f"Best Acc: {best_acc:.3f} | "
            f"Mean Acc: {np.mean(test_accs[-20:]):.3f} | "
            f"LR: {lr:.5f} | "
            f"Train Loss X: {train_loss_x:.3f} | "
            f"Train Loss U: {train_loss_u:.3f} "
        )

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])
