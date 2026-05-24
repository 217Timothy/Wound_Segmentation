"""Soft Dice loss (per-sample, then averaged)."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        target = target.float()
        probs = torch.sigmoid(logits)

        dims = (1, 2, 3)
        intersection = (probs * target).sum(dims)
        denom = probs.sum(dims) + target.sum(dims)

        score = (2.0 * intersection + self.smooth) / (denom + self.smooth)
        return (1.0 - score).mean()
