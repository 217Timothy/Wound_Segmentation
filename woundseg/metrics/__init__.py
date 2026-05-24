"""Segmentation evaluation metrics."""

from .functional import (
    all_metrics,
    dice_score,
    iou_score,
    precision_score,
    recall_score,
)

__all__ = [
    "all_metrics", "dice_score", "iou_score", "precision_score", "recall_score",
]
