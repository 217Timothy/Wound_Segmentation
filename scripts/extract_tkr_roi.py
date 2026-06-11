#!/usr/bin/env python3
"""Extract wound ROIs from TKR images using tkr-finetune-v1/exp2.

Pipeline per image:
  1. Load full-resolution image from data/raw/segmentation/TKR/images/
  2. Run model inference (at --image_size resolution) → binary mask
  3. Resize mask back to original image resolution
  4. Find wound bounding box in full-res mask → add padding → crop original image
  5. Save cropped ROI to <out_root>/<sheet_name>/

Sheet membership is determined by matching image stems against the filenames
already organised in tkr_roi_output_structure/ (the reference layout you provided).

Output layout:
    outputs/roi/tkr-finetune-v1/exp2/
        工作表1/IMG_XXXX.jpg
        07/IMG_XXXX.jpg
        08/IMG_XXXX.jpg
        09/IMG_XXXX.jpg
        10/IMG_XXXX.jpg
        11/IMG_XXXX.jpg

Usage:
    python scripts/extract_tkr_roi.py
    python scripts/extract_tkr_roi.py --padding 60 --sheets 07 08
    python scripts/extract_tkr_roi.py --ckpt outputs/checkpoints/tkr-finetune-v1/exp2/best.pt
"""

import argparse
import os
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

# ── config ───────────────────────────────────────────────────────────────────
SHEET_NAMES   = ["工作表1", "07", "08", "09", "10", "11"]
IMAGE_EXTS    = {".jpg", ".jpeg", ".png", ".bmp"}   # HEIC excluded (OpenCV can't read)
INFER_EXTS    = IMAGE_EXTS                           # same for raw images


def _clean(name: str) -> str:
    """Strip leading \x7f bytes that macOS sometimes adds to folder names."""
    return name.lstrip("\x7f")


# ── build stem → sheet mapping from reference layout ─────────────────────────
def build_stem_to_sheet(ref_root: Path) -> dict[str, str]:
    """Scan tkr_roi_output_structure/ and return {image_stem: sheet_name}."""
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


# ── ROI extraction ────────────────────────────────────────────────────────────
def _tight_crop(bgr: np.ndarray, mask_bin: np.ndarray, padding: int):
    """Apply mask to image (non-mask pixels → black), then crop to mask bbox.

    Returns (roi, bbox) or (None, None) if mask is empty.
    """
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

    # Zero out pixels outside the mask, then crop to bbox
    masked = bgr * mask_bin[:, :, np.newaxis]
    return masked[y1:y2, x1:x2], (x1, y1, x2, y2)


def extract_roi(
    bgr: np.ndarray,
    mask_full: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, str, tuple | None]:
    """Extract the masked wound region and crop tightly around it.

    Output pixels = original colour inside mask, black outside mask,
    then cropped to the mask's bounding box (+ padding) to remove black borders.

    Strategy:
      1. Primary  : use model segmentation mask.
      2. Fallback : black-background threshold mask (photos have pure-black backdrop).
      3. Last resort: return full image unchanged.

    Returns (roi_bgr, method, bbox).
    """
    img_h, img_w = bgr.shape[:2]

    # ── primary: model mask ───────────────────────────────────────────────
    model_bin = (mask_full > 0).astype(np.uint8)
    roi, bbox = _tight_crop(bgr, model_bin, padding)
    if roi is not None:
        return roi, "model", bbox

    # ── fallback: black-background removal ────────────────────────────────
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, fg_bin = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    fg_bin = cv2.morphologyEx(fg_bin, cv2.MORPH_CLOSE, kernel)

    roi, bbox = _tight_crop(bgr, fg_bin, padding)
    if roi is not None:
        return roi, "bg_removal", bbox

    # ── last resort ───────────────────────────────────────────────────────
    return bgr, "full", None


# ── main loop ─────────────────────────────────────────────────────────────────
def process(
    raw_images: list[Path],
    stem_to_sheet: dict[str, str],
    out_root: Path,
    model,
    device,
    transform,
    inf_size: int,
    padding: int,
    threshold: float,
    pp_kwargs: dict,
    target_sheets: list[str],
):
    counters: dict[str, dict] = {
        s: {"model": 0, "bg_removal": 0, "full": 0} for s in target_sheets
    }
    skipped_sheet = 0

    # Filter to only images that belong to target sheets
    to_process = [
        p for p in raw_images
        if stem_to_sheet.get(p.stem.lower()) in target_sheets
    ]
    skipped_sheet = len(raw_images) - len(to_process)

    for img_path in tqdm(to_process, desc="Extracting ROIs", unit="img"):
        stem  = img_path.stem.lower()
        sheet = stem_to_sheet[stem]

        # ── load ──────────────────────────────────────────────────────────
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            tqdm.write(f"  [warn] cannot read {img_path.name}")
            continue
        img_h, img_w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # ── inference at inference resolution ─────────────────────────────
        tensor   = transform(image=rgb)["image"]
        pred_inf = infer_one_image(model, tensor, device, threshold)
        pred_inf = postprocess_mask(pred_inf, **pp_kwargs)  # (H, W) 0/1

        # ── resize mask back to original image resolution ─────────────────
        pred_full = cv2.resize(
            pred_inf.astype(np.uint8),
            (img_w, img_h),
            interpolation=cv2.INTER_NEAREST,
        )

        # ── mask + crop ROI ───────────────────────────────────────────────
        roi, method, _ = extract_roi(bgr, pred_full, padding)

        # ── save ──────────────────────────────────────────────────────────
        out_dir = out_root / sheet
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(out_dir / (img_path.stem + ".jpg")),
            roi,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        counters[sheet][method] += 1

    return counters, skipped_sheet


def main():
    parser = argparse.ArgumentParser(description="Extract TKR wound ROIs")
    parser.add_argument("--ckpt",
        default="outputs/checkpoints/tkr-finetune-v1/exp2/best.pt")
    parser.add_argument("--raw_root",
        default="data/raw/segmentation/TKR/images",
        help="Folder with all raw TKR images (flat, no subfolders)")
    parser.add_argument("--ref_root",
        default="tkr_roi_output_structure",
        help="Reference layout folder used only for sheet membership lookup")
    parser.add_argument("--out_root",
        default="outputs/roi/tkr-finetune-v1/exp2")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--padding",    type=int, default=40,
        help="Pixels to pad around wound bounding box in original image (default: 40)")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--sheets",     nargs="+", default=None,
        help="Process only these sheet names, e.g. --sheets 07 08 工作表1")
    args = parser.parse_args()

    proj = Path(__file__).resolve().parents[1]
    ckpt_path = (Path(args.ckpt) if Path(args.ckpt).is_absolute()
                 else proj / args.ckpt)
    raw_root  = (Path(args.raw_root) if Path(args.raw_root).is_absolute()
                 else proj / args.raw_root)
    ref_root  = (Path(args.ref_root) if Path(args.ref_root).is_absolute()
                 else proj / args.ref_root)
    out_root  = (Path(args.out_root) if Path(args.out_root).is_absolute()
                 else proj / args.out_root)

    for p, label in [(ckpt_path, "checkpoint"), (raw_root, "raw images"),
                     (ref_root, "reference layout")]:
        if not p.exists():
            raise SystemExit(f"[error] {label} not found: {p}")

    target_sheets = args.sheets if args.sheets else SHEET_NAMES

    # ── load model ────────────────────────────────────────────────────────
    device   = get_device()
    fallback = {"model": "efficientunet", "encoder_name": "efficientnet-b3",
                "attention": "scse", "classes": 1}
    model, _ = load_inference_model(ckpt_path, device, fallback_cfg=fallback)
    print(f"[extract_roi] device={device}  ckpt={ckpt_path.relative_to(proj)}")

    transform = get_val_transforms((args.image_size, args.image_size))
    pp_kwargs = dict(blur_kernel=7, blur_sigma=0.0,
                     closing_kernel=7, open_kernel=0,
                     min_area=200, keep_largest=True)

    # ── build stem → sheet mapping ────────────────────────────────────────
    print("[extract_roi] Building sheet membership from reference layout …")
    stem_to_sheet = build_stem_to_sheet(ref_root)
    print(f"[extract_roi] {len(stem_to_sheet)} stems mapped across sheets")

    # ── collect raw images ────────────────────────────────────────────────
    raw_images = sorted(
        p for p in raw_root.iterdir()
        if p.is_file()
        and p.suffix.lower() in INFER_EXTS
        and not p.name.startswith(".")
    )
    print(f"[extract_roi] {len(raw_images)} raw images found in {raw_root.relative_to(proj)}")
    print(f"[extract_roi] Target sheets: {target_sheets}\n")

    # ── run ───────────────────────────────────────────────────────────────
    counters, skipped_sheet = process(
        raw_images=raw_images,
        stem_to_sheet=stem_to_sheet,
        out_root=out_root,
        model=model,
        device=device,
        transform=transform,
        inf_size=args.image_size,
        padding=args.padding,
        threshold=args.threshold,
        pp_kwargs=pp_kwargs,
        target_sheets=target_sheets,
    )

    # ── summary ───────────────────────────────────────────────────────────
    print("\n── Results ──────────────────────────────────────────────────────────")
    print(f"  {'Sheet':<10} {'model':>6} {'bg_rmv':>7} {'full':>6}")
    print(f"  {'-'*34}")
    t_model = t_bg = t_full = 0
    for sheet in target_sheets:
        c = counters[sheet]
        print(f"  {sheet:<10} {c['model']:>6} {c['bg_removal']:>7} {c['full']:>6}")
        t_model += c["model"]; t_bg += c["bg_removal"]; t_full += c["full"]
    print(f"  {'-'*34}")
    print(f"  {'TOTAL':<10} {t_model:>6} {t_bg:>7} {t_full:>6}")
    print(f"\n  model     = wound detected by segmentation model")
    print(f"  bg_rmv    = model mask empty → used black-background removal fallback")
    print(f"  full      = both methods failed → saved full image")
    print(f"  (skipped {skipped_sheet} images not belonging to target sheets)")
    print(f"\n[extract_roi] Output → {out_root.relative_to(proj)}")


if __name__ == "__main__":
    main()
