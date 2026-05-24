"""Tversky loss.

Tversky generalizes Dice by weighting false positives (`alpha`) and false
negatives (`beta`) separately. Setting beta > alpha penalizes missed wound
pixels harder, which helps recall on small lesions.

The score is computed per sample (over C, H, W) so that Focal-Tversky can
raise each sample's loss to a power before averaging.
"""

import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.4, beta: float = 0.6, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, target, reduce: bool = True):
        target = target.float()
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)

        dims = (1, 2, 3)  # keep the batch dimension
        tp = (probs * target).sum(dims)
        fp = (probs * (1 - target)).sum(dims)
        fn = ((1 - probs) * target).sum(dims)

        score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1.0 - score
        return loss.mean() if reduce else loss
