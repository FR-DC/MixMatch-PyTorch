import torch
import torch.nn as nn
import torch.nn.parallel


class WeightEMA(object):
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        ema_wgt_decay: float = 0.999,
        lr: float = 0.002,
    ):
        self.model = model
        self.ema_model = ema_model
        self.alpha = ema_wgt_decay
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * (1.0 - self.alpha))
                # customized weight decay
                param.mul_(1 - self.wd)
