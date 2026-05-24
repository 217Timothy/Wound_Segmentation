"""Combined losses: BCE paired with region-overlap losses.

BCE gives a stable pixel-wise gradient; the Dice/Tversky term optimizes the
overlap directly. Focal-Tversky additionally focuses on hard samples.
"""

import torch
import torch.nn as nn

from .dice import DiceLoss
from .tversky import TverskyLoss


class BCEDiceLoss(nn.Module):
    """`bce_weight * BCE + (1 - bce_weight) * Dice`."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits, target):
        return (self.bce_weight * self.bce(logits, target.float())
                + (1 - self.bce_weight) * self.dice(logits, target))


class BCETverskyLoss(nn.Module):
    """`bce_weight * BCE + (1 - bce_weight) * Tversky`."""

    def __init__(self, alpha: float = 0.4, beta: float = 0.6,
                 bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits, target):
        return (self.bce_weight * self.bce(logits, target.float())
                + (1 - self.bce_weight) * self.tversky(logits, target))


class FocalTverskyLoss(nn.Module):
    """`mean( TverskyLoss_per_sample ** gamma )`.

    gamma > 1 down-weights easy samples (low loss) and focuses training on
    hard ones.
    """

    def __init__(self, alpha: float = 0.4, beta: float = 0.6,
                 gamma: float = 1.33, smooth: float = 1e-6):
        super().__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, logits, target):
        per_sample = self.tversky(logits, target, reduce=False)
        return per_sample.pow(self.gamma).mean()
