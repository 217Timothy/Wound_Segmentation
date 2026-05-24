#!/usr/bin/env python3
"""Evaluate a checkpoint on a split and report per-class Dice / IoU.

Per-class Dice is printed prominently because the project goal is >=0.80
Dice for every wound class. Results are also saved as metrics.json.

Example:
    python scripts/evaluate.py --config configs/evaluate.yaml
    python scripts/evaluate.py --config configs/evaluate.yaml --split test
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from woundseg.config import load_config
from woundseg.data import WoundSegmentationDataset, get_val_transforms
from woundseg.engine import load_inference_model
from woundseg.metrics import all_metrics
from woundseg.utils import get_device

EVALUATE_DEFAULTS = {
    "version": "exp",
    "run_name": "run1",
    "ckpt": None,
    "data_root": "data/processed/wound",
    "datasets": [],
    "split": "val",
    "image_size": 512,
    "threshold": 0.5,
    "num_workers": 4,
    "out_root": "outputs",
    "target_dice": 0.80,
    # fallback architecture for legacy checkpoints
    "model": "efficientunet",
    "encoder_name": "efficientnet-b3",
    "attention": "scse",
}


@torch.no_grad()
def evaluate_dataset(model, loader, device, threshold):
    """Average every metric over one dataset's loader (batch size 1)."""
    keys = ("dice", "iou", "recall", "precision")
    totals = {k: 0.0 for k in keys}
    n = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        preds = (torch.sigmoid(model(imgs)) > threshold).float().cpu()
        for i in range(preds.shape[0]):
            sample = all_metrics(preds[i:i + 1], masks[i:i + 1])
            for k in keys:
                totals[k] += sample[k]
            n += 1
    return {k: (totals[k] / n if n else 0.0) for k in keys}, n


def main():
    cfg = load_config(EVALUATE_DEFAULTS)
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")

    device = get_device()
    out_root = Path(cfg.out_root)
    ckpt_path = Path(cfg.ckpt) if cfg.ckpt else (
        out_root / "checkpoints" / cfg.version / cfg.run_name / "best.pt")
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    fallback = {"model": cfg.model, "encoder_name": cfg.encoder_name,
                "attention": cfg.attention, "classes": 1}
    model, _ = load_inference_model(ckpt_path, device, fallback_cfg=fallback)
    print(f"[eval] device={device}  checkpoint={ckpt_path}  split={cfg.split}\n")

    transform = get_val_transforms((cfg.image_size, cfg.image_size))
    report = {}
    for ds in cfg.datasets:
        dataset = WoundSegmentationDataset(cfg.data_root, [ds], cfg.split, transform)
        if len(dataset) == 0:
            print(f"[eval][warn] no samples for {ds} ({cfg.split}); skipping")
            continue
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers)
        metrics, n = evaluate_dataset(model, loader, device, cfg.threshold)
        report[ds] = {**metrics, "samples": n}

    if not report:
        raise SystemExit("Nothing evaluated.")

    keys = ("dice", "iou", "recall", "precision")
    report["all"] = {k: sum(report[d][k] for d in report) / len(report) for k in keys}

    print(f"{'dataset':<16}{'dice':>9}{'iou':>9}{'recall':>9}{'precision':>11}   status")
    print("-" * 70)
    for ds, m in report.items():
        status = "" if ds == "all" else (
            "OK" if m["dice"] >= cfg.target_dice else "BELOW TARGET")
        print(f"{ds:<16}{m['dice']:>9.4f}{m['iou']:>9.4f}"
              f"{m['recall']:>9.4f}{m['precision']:>11.4f}   {status}")

    below = [d for d, m in report.items()
             if d != "all" and m["dice"] < cfg.target_dice]
    print()
    if below:
        print(f"[eval] classes below {cfg.target_dice:.2f} Dice: {', '.join(below)}")
    else:
        print(f"[eval] all classes reached the {cfg.target_dice:.2f} Dice target.")

    out_dir = out_root / "runs" / cfg.version / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{cfg.split}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[eval] saved {out_path}")


if __name__ == "__main__":
    main()
