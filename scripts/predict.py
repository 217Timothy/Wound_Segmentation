#!/usr/bin/env python3
"""Run inference on folders of images and save masks + visualizations.

Input layout (per dataset):
    <in_root>/<dataset>/images/*.png   (masks/ optional, used for GT overlay)

Output layout:
    <out_root>/predictions/<version>/<run_name>/<dataset>/*.png
    <out_root>/visualizations/<version>/<run_name>/overlay|combine/<dataset>/*.png

附帶計時功能：印硬體資訊、每張圖延遲、FPS、總時間，寫 timing.json。

Example:
    python scripts/predict.py --config configs/predict.yaml
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from woundseg.config import load_config
from woundseg.data import get_val_transforms
from woundseg.engine import infer_one_image, load_inference_model
from woundseg.postprocess import postprocess_mask
from woundseg.utils import (LatencyMeter, get_device, make_combine, make_overlay,
                            make_overlay_with_gt, print_system_info, sync_cuda)

PREDICT_DEFAULTS = {
    "version": "exp",
    "run_name": "run1",
    "ckpt": None,                       # explicit checkpoint; overrides version/run_name
    "datasets": [],
    "in_root": "data/raw/segmentation",
    "out_root": "outputs",
    "image_size": 512,
    "threshold": 0.5,
    "postprocess": True,
    "blur_kernel": 7,
    "blur_sigma": 0.0,
    "closing_kernel": 7,
    "open_kernel": 0,
    "min_area": 200,
    "keep_largest": True,
    "warmup_iters": 3,                   # 計時前 warm-up
    # fallback architecture for legacy checkpoints without 'model_cfg'
    "model": "efficientunet",
    "encoder_name": "efficientnet-b3",
    "attention": "scse",
}

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def collect_images(dataset_dir: Path) -> list[Path]:
    """Return sorted image paths from <dataset_dir>/images or <dataset_dir>."""
    search = dataset_dir / "images"
    if not search.is_dir():
        search = dataset_dir
    return sorted(p for p in search.iterdir()
                  if p.suffix.lower() in _IMAGE_EXTS)


def find_gt_mask(dataset_dir: Path, stem: str, size: int):
    """Look for a matching ground-truth mask; return it resized or None."""
    mask_dir = dataset_dir / "masks"
    for ext in _IMAGE_EXTS:
        path = mask_dir / f"{stem}{ext}"
        if path.exists():
            m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    return None


def main():
    sys_info = print_system_info()

    cfg = load_config(PREDICT_DEFAULTS)
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
    print(f"[predict] device={device}  checkpoint={ckpt_path}")

    size = cfg.image_size
    transform = get_val_transforms((size, size))
    pp_kwargs = dict(blur_kernel=cfg.blur_kernel, blur_sigma=cfg.blur_sigma,
                     closing_kernel=cfg.closing_kernel, open_kernel=cfg.open_kernel,
                     min_area=cfg.min_area, keep_largest=cfg.keep_largest)

    # 各類別 + 整體計時
    per_class_timing: dict[str, dict] = {}
    overall_meter_pred = LatencyMeter()   # 純前向
    overall_meter_full = LatencyMeter()   # 含後處理 + 視覺化寫檔

    t_total_start = time.perf_counter()

    for ds in cfg.datasets:
        dataset_dir = Path(cfg.in_root) / ds
        images = collect_images(dataset_dir)
        print(f"[predict] {ds}: {len(images)} images")

        pred_dir = out_root / "predictions" / cfg.version / cfg.run_name / ds
        overlay_dir = out_root / "visualizations" / cfg.version / cfg.run_name / "overlay" / ds
        combine_dir = out_root / "visualizations" / cfg.version / cfg.run_name / "combine" / ds
        for d in (pred_dir, overlay_dir, combine_dir):
            d.mkdir(parents=True, exist_ok=True)

        ds_meter_pred = LatencyMeter()
        ds_meter_full = LatencyMeter()

        # warm-up
        warm_n = min(cfg.warmup_iters, len(images))
        for img_path in images[:warm_n]:
            bgr = cv2.imread(str(img_path))
            if bgr is None: continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(image=rgb)["image"]
            _ = infer_one_image(model, tensor, device, cfg.threshold)
        sync_cuda()

        # 正式跑（全部，包含 warm-up 那幾張，因為要產出結果）
        for img_path in images:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(f"[predict][warn] unreadable image: {img_path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = transform(image=rgb)["image"]

            # 純前向計時
            with ds_meter_pred:
                pred = infer_one_image(model, tensor, device, cfg.threshold)
            overall_meter_pred.times.append(ds_meter_pred.times[-1])

            # 含後處理 + 視覺化計時
            with ds_meter_full:
                if cfg.postprocess:
                    pred = postprocess_mask(pred, **pp_kwargs)
                stem = img_path.stem
                cv2.imwrite(str(pred_dir / f"{stem}.png"), pred * 255)

                vis = cv2.resize(bgr, (size, size))
                gt = find_gt_mask(dataset_dir, stem, size)
                overlay = (make_overlay_with_gt(vis, pred, gt) if gt is not None
                           else make_overlay(vis, pred))
                cv2.imwrite(str(overlay_dir / f"{stem}.png"), overlay)
                cv2.imwrite(str(combine_dir / f"{stem}.png"), make_combine(vis, pred, gt))
            overall_meter_full.times.append(ds_meter_full.times[-1])

        per_class_timing[ds] = {
            "pred_only": ds_meter_pred.summary(),
            "full_pipeline": ds_meter_full.summary(),
        }
        p = ds_meter_pred.summary()
        print(f"    avg latency={p.get('mean_ms', 0)} ms  FPS={p.get('fps', 0)}")

    total_seconds = time.perf_counter() - t_total_start

    print()
    print(overall_meter_pred.summary_text())
    print(f"\n  含後處理/寫檔的整體 wall-clock：{total_seconds:.2f}s")

    # 寫 JSON
    out_dir = out_root / "runs" / cfg.version / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "system": sys_info,
        "config": {
            "version": cfg.version,
            "run_name": cfg.run_name,
            "datasets": cfg.datasets,
            "image_size": cfg.image_size,
            "threshold": cfg.threshold,
            "postprocess": cfg.postprocess,
            "warmup_iters": cfg.warmup_iters,
        },
        "timing": {
            "wall_clock_seconds": round(total_seconds, 3),
            "overall_pred_only": overall_meter_pred.summary(),
            "overall_full_pipeline": overall_meter_full.summary(),
            "per_class": per_class_timing,
        },
    }
    out_file = out_dir / "predict_timing.json"
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[predict] timing JSON：{out_file}")
    print("[predict] done.")


if __name__ == "__main__":
    main()
