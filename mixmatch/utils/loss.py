import numpy as np
import torch
import torch.nn.parallel
from torch.nn.functional import cross_entropy


def linear_rampup(current: float, rampup_length: int):
    if rampup_length == 0:
        return 1.0
    return np.clip(current / rampup_length, 0, 1)


class SemiLoss(object):
    def __call__(
        self,
        x_lbl: torch.Tensor,
        y_lbl: torch.Tensor,
        x_unl: torch.Tensor,
        y_unl: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probs_u = torch.softmax(x_unl, dim=1)

        l_x = cross_entropy(x_lbl, y_lbl)
        # TODO: Not sure why this is different from MSELoss
        #  It's likely not a big deal, but it's worth investigating if we have
        #  too much time on our hands
        l_u = torch.mean((probs_u - y_unl) ** 2)
        return l_x, l_u
