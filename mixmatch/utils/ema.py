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
        self.model = model
        self.ema_model = ema_model
        self.ema_lr = ema_lr
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(1 - self.ema_lr)
                ema_param.add_(param * self.ema_lr)
