#!/usr/bin/env python3
"""Debug version of extract_tkr_roi.py.

Saves a side-by-side composite  [ mask | roi ]  for each processed image so
you can visually verify the mask quality and crop result.

Output layout:
    outputs/debug/tkr-finetune-v1/exp2/
        工作表1/IMG_XXXX.jpg   ← [mask | roi] composite
        07/IMG_XXXX.jpg
        ...

Usage:
    # First 10 images per sheet (default)
    python scripts/debug_tkr_roi.py

    # First 30 images per sheet
    python scripts/debug_tkr_roi.py --n_per_sheet 30

    # All images (slow)
    python scripts/debug_tkr_roi.py --n_per_sheet 0

    # Only specific sheets
    python scripts/debug_tkr_roi.py --sheets 07 08 --n_per_sheet 20
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from tqdm import tqdm

from woundseg.data import get_val_transforms
from woundseg.engine import infer_one_image, load_inference_model
from woundseg.postprocess import postprocess_mask
from woundseg.utils import get_device

# ── shared helpers (copy-free import from extract_tkr_roi) ───────────────────
SHEET_NAMES = ["工作表1", "07", "08", "09", "10", "11"]
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}
DEBUG_H     = 600   # height of each panel in the composite (pixels)


def _clean(name: str) -> str:
    return name.lstrip("\x7f")


def build_stem_to_sheet(ref_root: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in ref_root.iterdir():
        if not entry.is_dir():
            continue
        sheet = _clean(entry.name)
        if sheet not in SHEET_NAMES:
            continue
        for f in entry.iterdir():
            if f.is_file() and not f.name.startswith("."):
                mapping[f.stem.lower()] = sheet
    return mapping


def _tight_crop(bgr: np.ndarray, mask_bin: np.ndarray, padding: int):
    img_h, img_w = bgr.shape[:2]
    contours, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img_w, x + w + padding)
    y2 = min(img_h, y + h + padding)
    masked = bgr * mask_bin[:, :, np.newaxis]
    return masked[y1:y2, x1:x2], (x1, y1, x2, y2)


def extract_roi(bgr, mask_full, padding):
    img_h, img_w = bgr.shape[:2]

    model_bin = (mask_full > 0).astype(np.uint8)
    roi, bbox = _tight_crop(bgr, model_bin, padding)
    if roi is not None:
        return roi, "model", bbox

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    roi, bbox = _tight_crop(bgr, fg, padding)
    if roi is not None:
        return roi, "bg_removal", bbox

    return bgr, "full", None


# ── composite builder ─────────────────────────────────────────────────────────
def make_composite(
    bgr_orig: np.ndarray,
    mask_inf: np.ndarray,     # inference-resolution mask, values 0/1
    mask_full: np.ndarray,    # full-resolution mask, values 0/1
    roi: np.ndarray,
    method: str,
    bbox,
    target_h: int = DEBUG_H,
) -> np.ndarray:
    """Build a  [ mask_panel | roi_panel ]  side-by-side image.

    mask_panel  = original image resized to target_h with:
                    • colour mask overlay (semi-transparent green)
                    • red bounding-box rectangle
                    • method label

    roi_panel   = ROI crop resized to target_h, with method label
    """
    img_h, img_w = bgr_orig.shape[:2]

    # ── left panel: mask overlay on original ─────────────────────────────
    vis = bgr_orig.copy()

    # green overlay where mask = 1
    green = np.zeros_like(vis)
    green[mask_full > 0] = (0, 220, 0)
    vis = cv2.addWeighted(vis, 1.0, green, 0.45, 0)

    # red bbox rectangle
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), max(2, img_w // 300))

    # method label
    label_color = (0, 200, 0) if method == "model" else (0, 180, 255)
    cv2.putText(vis, f"mask [{method}]", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, f"mask [{method}]", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, label_color, 2, cv2.LINE_AA)

    # resize to target_h
    scale  = target_h / img_h
    left_w = max(1, int(img_w * scale))
    left   = cv2.resize(vis, (left_w, target_h), interpolation=cv2.INTER_AREA)

    # ── right panel: ROI crop ─────────────────────────────────────────────
    r_h, r_w = roi.shape[:2]
    r_scale   = target_h / r_h
    right_w   = max(1, int(r_w * r_scale))
    right     = cv2.resize(roi, (right_w, target_h), interpolation=cv2.INTER_AREA)

    # label
    cv2.putText(right, "roi", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(right, "roi", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    # ── divider ───────────────────────────────────────────────────────────
    divider = np.full((target_h, 4, 3), 80, dtype=np.uint8)

    return np.concatenate([left, divider, right], axis=1)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Debug TKR ROI extraction")
    parser.add_argument("--ckpt",
        default="outputs/checkpoints/tkr-finetune-v1/exp2/best.pt")
    parser.add_argument("--raw_root",
        default="data/raw/segmentation/TKR/images")
    parser.add_argument("--ref_root",
        default="tkr_roi_output_structure")
    parser.add_argument("--out_root",
        default="outputs/debug/tkr-finetune-v1/exp2")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--padding",    type=int, default=40)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--n_per_sheet", type=int, default=10,
        help="Images to process per sheet (0 = all, default: 10)")
    parser.add_argument("--sheets", nargs="+", default=None)
    args = parser.parse_args()

    proj      = Path(__file__).resolve().parents[1]
    ckpt_path = proj / args.ckpt
    raw_root  = proj / args.raw_root
    ref_root  = proj / args.ref_root
    out_root  = proj / args.out_root

    for p, label in [(ckpt_path, "checkpoint"), (raw_root, "raw images"),
                     (ref_root, "reference layout")]:
        if not p.exists():
            raise SystemExit(f"[error] {label} not found: {p}")

    target_sheets = args.sheets or SHEET_NAMES

    # ── load model ────────────────────────────────────────────────────────
    device   = get_device()
    fallback = {"model": "efficientunet", "encoder_name": "efficientnet-b3",
                "attention": "scse", "classes": 1}
    model, _ = load_inference_model(ckpt_path, device, fallback_cfg=fallback)
    print(f"[debug_roi] device={device}  ckpt={ckpt_path.relative_to(proj)}")

    transform = get_val_transforms((args.image_size, args.image_size))
    pp_kwargs = dict(blur_kernel=7, blur_sigma=0.0,
                     closing_kernel=7, open_kernel=0,
                     min_area=200, keep_largest=True)

    # ── build sheet mapping ───────────────────────────────────────────────
    stem_to_sheet = build_stem_to_sheet(ref_root)
    print(f"[debug_roi] {len(stem_to_sheet)} stems mapped")

    # ── collect images, grouped by sheet ─────────────────────────────────
    all_images = sorted(
        p for p in raw_root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )

    sheet_images: dict[str, list[Path]] = {s: [] for s in target_sheets}
    for img_path in all_images:
        sheet = stem_to_sheet.get(img_path.stem.lower())
        if sheet in target_sheets:
            sheet_images[sheet].append(img_path) # type: ignore

    # Limit per sheet
    to_process: list[tuple[str, Path]] = []
    for sheet in target_sheets:
        imgs = sheet_images[sheet]
        if args.n_per_sheet > 0:
            imgs = imgs[:args.n_per_sheet]
        to_process.extend((sheet, p) for p in imgs)

    n_total = len(to_process)
    n_lim   = args.n_per_sheet if args.n_per_sheet > 0 else "all"
    print(f"[debug_roi] {n_total} images to process "
          f"({n_lim} per sheet × {len(target_sheets)} sheets)\n")

    # ── process ───────────────────────────────────────────────────────────
    method_counts = {"model": 0, "bg_removal": 0, "full": 0}

    for sheet, img_path in tqdm(to_process, desc="Debug composites", unit="img"):
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            tqdm.write(f"  [warn] cannot read {img_path.name}")
            continue

        img_h, img_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # inference
        tensor   = transform(image=rgb)["image"]
        pred_inf = infer_one_image(model, tensor, device, args.threshold)
        pred_inf = postprocess_mask(pred_inf, **pp_kwargs) # type: ignore

        # resize mask to original resolution
        pred_full = cv2.resize(
            pred_inf.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # extract ROI
        roi, method, bbox = extract_roi(bgr, pred_full, args.padding)
        method_counts[method] += 1

        # build composite
        composite = make_composite(
            bgr_orig  = bgr,
            mask_inf  = pred_inf,
            mask_full = pred_full,
            roi       = roi,
            method    = method,
            bbox      = bbox,
            target_h  = DEBUG_H,
        )

        # save
        out_dir = out_root / sheet
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_dir / (img_path.stem + ".jpg")),
            composite,
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )

    # ── summary ───────────────────────────────────────────────────────────
    print(f"\n── Debug Summary ───────────────────────────────────────────────")
    print(f"  model      (segmentation mask used)  : {method_counts['model']:4d}")
    print(f"  bg_removal (black-bg fallback used)  : {method_counts['bg_removal']:4d}")
    print(f"  full       (fallback failed, orig)   : {method_counts['full']:4d}")
    print(f"\n  Composites → {out_root.relative_to(proj)}/")
    print(f"  Format: [ original+mask_overlay+bbox | roi_crop ]")


if __name__ == "__main__":
    main()
