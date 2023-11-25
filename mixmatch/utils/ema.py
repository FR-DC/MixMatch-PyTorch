import torch
import torch.nn as nn
import torch.nn.parallel


class WeightEMA:
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        ema_lr: float,
    ):
        self.ema_lr = ema_lr
        self.model = model
        self.ema_model = ema_model

    def step(self):
        for param, ema_param in zip(self.model.parameters(),
                                    self.ema_model.parameters()):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(1 - self.ema_lr)
                ema_param.add_(param * self.ema_lr)
