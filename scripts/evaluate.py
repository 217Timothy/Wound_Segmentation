#!/usr/bin/env python3
"""Evaluate a checkpoint on a split and report per-class Dice / IoU.

Per-class Dice is printed prominently because the project goal is >=0.80
Dice for every wound class. Results are also saved as metrics.json.

附帶計時功能：印硬體資訊、量單張前向延遲、總時間、FPS。

Example:
    python scripts/evaluate.py --config configs/evaluate.yaml
    python scripts/evaluate.py --config configs/evaluate.yaml --split test
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from woundseg.config import load_config
from woundseg.data import WoundSegmentationDataset, get_val_transforms
from woundseg.engine import load_inference_model
from woundseg.metrics import all_metrics
from woundseg.utils import LatencyMeter, get_device, print_system_info, sync_cuda

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
    "warmup_iters": 3,  # 計時前 warm-up，比較公平
    # fallback architecture for legacy checkpoints
    "model": "efficientunet",
    "encoder_name": "efficientnet-b3",
    "attention": "scse",
}


@torch.no_grad()
def evaluate_dataset(model, loader, device, threshold, meter: LatencyMeter):
    """Average every metric over one dataset's loader (batch size 1)。
    同時用 meter 計每張前向時間（不含 DataLoader IO）。
    """
    keys = ("dice", "iou", "recall", "precision")
    totals = {k: 0.0 for k in keys}
    n = 0
    for imgs, masks, _ in loader:
        imgs = imgs.to(device)
        with meter:
            preds = (torch.sigmoid(model(imgs)) > threshold).float()
        preds = preds.cpu()
        for i in range(preds.shape[0]):
            sample = all_metrics(preds[i:i + 1], masks[i:i + 1])
            for k in keys:
                totals[k] += sample[k]
            n += 1
    return {k: (totals[k] / n if n else 0.0) for k in keys}, n


def main():
    sys_info = print_system_info()

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
    model.eval()
    print(f"[eval] device={device}  checkpoint={ckpt_path}  split={cfg.split}\n")

    transform = get_val_transforms((cfg.image_size, cfg.image_size))
    report = {}
    per_class_timing: dict = {}
    overall_meter = LatencyMeter()

    t_total_start = time.perf_counter()

    for ds in cfg.datasets:
        dataset = WoundSegmentationDataset(cfg.data_root, [ds], cfg.split, transform)
        if len(dataset) == 0:
            print(f"[eval][warn] no samples for {ds} ({cfg.split}); skipping")
            continue
        loader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers)

        # warm-up（不算進統計）
        warm_n = min(cfg.warmup_iters, len(dataset))
        if warm_n > 0:
            with torch.no_grad():
                for i, (imgs, _, _) in enumerate(loader):
                    if i >= warm_n:
                        break
                    _ = model(imgs.to(device))
            sync_cuda()

        # 正式計時 + 算指標
        meter = LatencyMeter()
        metrics, n = evaluate_dataset(model, loader, device, cfg.threshold, meter)
        report[ds] = {**metrics, "samples": n}
        per_class_timing[ds] = meter.summary()
        # 合進 overall（直接 extend times）
        overall_meter.times.extend(meter.times)

    if not report:
        raise SystemExit("Nothing evaluated.")

    total_seconds = time.perf_counter() - t_total_start

    keys = ("dice", "iou", "recall", "precision")
    report["all"] = {k: sum(report[d][k] for d in report) / len(report) for k in keys}

    # ===== 印準確率（原本就有） =====
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

    # ===== 印計時（新增） =====
    print()
    print(overall_meter.summary_text())
    print(f"\n  含 IO/DataLoader 總時間：{total_seconds:.2f}s")

    # 存報告（含計時）
    out_dir = out_root / "runs" / cfg.version / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"metrics_{cfg.split}.json"
    final = {
        "system": sys_info,
        "config": {
            "version": cfg.version,
            "run_name": cfg.run_name,
            "split": cfg.split,
            "image_size": cfg.image_size,
            "threshold": cfg.threshold,
        },
        "metrics": report,
        "timing": {
            "wall_clock_seconds": round(total_seconds, 3),
            "overall_inference": overall_meter.summary(),
            "per_class_inference": per_class_timing,
        },
    }
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] saved {out_path}")


if __name__ == "__main__":
    main()
