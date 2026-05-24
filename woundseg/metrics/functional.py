"""Evaluation metrics for binary segmentation.

All functions take already-binarized tensors (values in {0, 1}) of identical
shape and return a Python float. They flatten every dimension, so for
per-sample metrics pass one sample at a time.
"""

_EPS = 1e-6


def _flatten(pred, target):
    return pred.reshape(-1).float(), target.reshape(-1).float()


def dice_score(pred, target) -> float:
    """Dice coefficient = 2|A∩B| / (|A| + |B|)."""
    p, t = _flatten(pred, target)
    intersection = (p * t).sum()
    return ((2.0 * intersection + _EPS) / (p.sum() + t.sum() + _EPS)).item()


def iou_score(pred, target) -> float:
    """Intersection over Union = |A∩B| / |A∪B|."""
    p, t = _flatten(pred, target)
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return ((intersection + _EPS) / (union + _EPS)).item()


def recall_score(pred, target) -> float:
    """Recall = TP / (TP + FN)."""
    p, t = _flatten(pred, target)
    tp = (p * t).sum()
    fn = ((1 - p) * t).sum()
    return ((tp + _EPS) / (tp + fn + _EPS)).item()


def precision_score(pred, target) -> float:
    """Precision = TP / (TP + FP)."""
    p, t = _flatten(pred, target)
    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    return ((tp + _EPS) / (tp + fp + _EPS)).item()


def all_metrics(pred, target) -> dict:
    """Return every metric at once as a dict."""
    return {
        "dice": dice_score(pred, target),
        "iou": iou_score(pred, target),
        "recall": recall_score(pred, target),
        "precision": precision_score(pred, target),
    }
