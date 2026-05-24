"""Validation loop with per-dataset (per-class) metrics."""

from collections import defaultdict

import torch
from tqdm import tqdm

from ..metrics import all_metrics

_KEYS = ("dice", "iou", "recall", "precision")


@torch.no_grad()
def validate(model, loader, loss_func, device) -> dict:
    """Evaluate `model` and return loss plus overall and per-dataset metrics.

    Returns a dict with:
        loss          mean validation loss
        dice/iou/...  micro-average over every sample
        mean_dice     macro-average of per-dataset Dice (the per-class goal)
        per_dataset   {dataset_name: {dice, iou, recall, precision}}

    Metrics are computed one sample at a time, so any batch size is correct.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    stats = defaultdict(lambda: {"n": 0, **{k: 0.0 for k in _KEYS}})

    for imgs, masks, names in tqdm(loader, desc="validate"):
        imgs = imgs.to(device)
        masks = masks.to(device)

        logits = model(imgs)
        total_loss += loss_func(logits, masks).item()
        n_batches += 1

        preds = (torch.sigmoid(logits) > 0.5).float()
        for i in range(preds.shape[0]):
            sample = all_metrics(preds[i:i + 1], masks[i:i + 1])
            bucket = stats[names[i]]
            bucket["n"] += 1
            for k in _KEYS:
                bucket[k] += sample[k]

    datasets = sorted(stats)
    per_dataset = {
        ds: {k: stats[ds][k] / stats[ds]["n"] for k in _KEYS}
        for ds in datasets
    }

    n_total = sum(stats[ds]["n"] for ds in datasets) or 1
    micro = {k: sum(stats[ds][k] for ds in datasets) / n_total for k in _KEYS}
    macro_dice = (sum(per_dataset[ds]["dice"] for ds in datasets) / len(datasets)
                  if datasets else 0.0)

    return {
        "loss": total_loss / max(n_batches, 1),
        **micro,
        "mean_dice": macro_dice,
        "per_dataset": per_dataset,
    }
