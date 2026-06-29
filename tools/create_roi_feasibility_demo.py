"""Create ROI feasibility panels from predicted wound masks.

This report helper turns segmentation predictions into bbox/crop examples for
the original "segmentation -> ROI -> classification" direction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass
class RoiRecord:
    dataset: str
    stem: str
    image_path: str
    pred_path: str
    gt_path: str
    panel_path: str
    crop_path: str
    status: str
    iou: float
    pred_area_ratio: float
    bbox: str


def find_image(folder: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_mask(path: Path, shape: tuple[int, int] | None = None) -> np.ndarray | None:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if shape is not None and mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def mask_iou(pred: np.ndarray, gt: np.ndarray | None) -> float:
    if gt is None:
        return 0.0
    intersection = np.logical_and(pred > 0, gt > 0).sum()
    union = np.logical_or(pred > 0, gt > 0).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def expand_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
    margin_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1 + 1)
    bh = max(1, y2 - y1 + 1)
    mx = int(round(bw * margin_ratio))
    my = int(round(bh * margin_ratio))
    return (
        max(0, x1 - mx),
        max(0, y1 - my),
        min(width - 1, x2 + mx),
        min(height - 1, y2 + my),
    )


def classify_status(pred: np.ndarray, iou: float) -> str:
    area_ratio = float(np.count_nonzero(pred)) / float(pred.size)
    if area_ratio == 0:
        return "failed_empty"
    if iou >= 0.5:
        return "good"
    if iou >= 0.2:
        return "acceptable"
    return "failed_low_iou"


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    out = image.copy()
    colored = np.zeros_like(out)
    colored[:, :] = color
    return np.where(mask[..., None] > 0, cv2.addWeighted(out, 0.55, colored, 0.45, 0), out)


def label_panel(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    return out


def resize_square(image: np.ndarray, size: int = 320) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    y = (size - nh) // 2
    x = (size - nw) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def make_panel(
    *,
    image: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    bbox: tuple[int, int, int, int] | None,
    status: str,
    iou: float,
) -> np.ndarray:
    original = image.copy()
    gt_vis = image.copy()
    if gt_mask is not None:
        gt_vis = overlay_mask(gt_vis, gt_mask, (0, 180, 0))
    pred_vis = overlay_mask(image, pred_mask, (0, 0, 255))
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(pred_vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
        crop = image[y1 : y2 + 1, x1 : x2 + 1]
    else:
        crop = np.full_like(image, 235)

    panels = [
        label_panel(resize_square(original), "Original"),
        label_panel(resize_square(gt_vis), "GT mask"),
        label_panel(resize_square(pred_vis), f"Pred ROI | {status} | IoU {iou:.2f}"),
        label_panel(resize_square(crop), "ROI crop"),
    ]
    return np.concatenate(panels, axis=1)


def process_dataset(
    *,
    dataset: str,
    raw_root: Path,
    pred_root: Path,
    out_root: Path,
    margin_ratio: float,
) -> list[RoiRecord]:
    image_dir = raw_root / dataset / "images"
    mask_dir = raw_root / dataset / "masks"
    pred_dir = pred_root / dataset
    panel_dir = out_root / "all_panels" / dataset
    crop_dir = out_root / "all_crops" / dataset
    panel_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)

    records: list[RoiRecord] = []
    for pred_path in sorted(pred_dir.glob("*.png")):
        stem = pred_path.stem
        image_path = find_image(image_dir, stem)
        gt_path = find_image(mask_dir, stem)
        if image_path is None:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        pred_mask = read_mask(pred_path, (height, width))
        gt_mask = read_mask(gt_path, (height, width)) if gt_path is not None else None
        if pred_mask is None:
            continue

        iou = mask_iou(pred_mask, gt_mask)
        status = classify_status(pred_mask, iou)
        bbox = bbox_from_mask(pred_mask)
        expanded = expand_bbox(bbox, width, height, margin_ratio) if bbox else None
        panel = make_panel(
            image=image,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            bbox=expanded,
            status=status,
            iou=iou,
        )
        panel_path = panel_dir / f"{stem}.png"
        cv2.imwrite(str(panel_path), panel)

        if expanded is not None:
            x1, y1, x2, y2 = expanded
            crop = image[y1 : y2 + 1, x1 : x2 + 1]
        else:
            crop = np.full((256, 256, 3), 235, dtype=np.uint8)
        crop_path = crop_dir / f"{stem}.png"
        cv2.imwrite(str(crop_path), crop)

        area_ratio = float(np.count_nonzero(pred_mask)) / float(pred_mask.size)
        bbox_text = "" if expanded is None else ",".join(str(v) for v in expanded)
        records.append(
            RoiRecord(
                dataset=dataset,
                stem=stem,
                image_path=str(image_path),
                pred_path=str(pred_path),
                gt_path=str(gt_path) if gt_path else "",
                panel_path=str(panel_path),
                crop_path=str(crop_path),
                status=status,
                iou=iou,
                pred_area_ratio=area_ratio,
                bbox=bbox_text,
            )
        )
    return records


def choose_selected(records: list[RoiRecord], per_status: int) -> list[RoiRecord]:
    selected: list[RoiRecord] = []
    by_dataset = sorted({r.dataset for r in records})
    for dataset in by_dataset:
        ds_records = [r for r in records if r.dataset == dataset]
        good = sorted(
            (r for r in ds_records if r.status == "good"),
            key=lambda r: r.iou,
            reverse=True,
        )[:per_status]
        acceptable = sorted(
            (r for r in ds_records if r.status == "acceptable"),
            key=lambda r: r.iou,
            reverse=True,
        )[:per_status]
        failed = sorted(
            (r for r in ds_records if r.status.startswith("failed")),
            key=lambda r: r.iou,
        )[:per_status]
        selected.extend(good + acceptable + failed)
    return selected


def copy_selected(selected: list[RoiRecord], out_root: Path) -> None:
    selected_dir = out_root / "selected_panels"
    selected_dir.mkdir(parents=True, exist_ok=True)
    for old_file in selected_dir.glob("*.png"):
        old_file.unlink()
    for idx, record in enumerate(selected, start=1):
        src = Path(record.panel_path)
        dst = selected_dir / f"{idx:02d}_{record.dataset}_{record.status}_{record.stem}.png"
        cv2.imwrite(str(dst), cv2.imread(str(src)))


def write_csv(records: list[RoiRecord], out_root: Path) -> None:
    path = out_root / "roi_metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RoiRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            row = record.__dict__.copy()
            row["iou"] = f"{record.iou:.4f}"
            row["pred_area_ratio"] = f"{record.pred_area_ratio:.6f}"
            writer.writerow(row)


def write_summary(records: list[RoiRecord], selected: list[RoiRecord], out_root: Path) -> None:
    lines = [
        "# ROI Classification Feasibility Demo",
        "",
        "This demo uses the current wound segmentation model to produce predicted masks,",
        "bbox ROIs, and wound crops for the original ROI-based classification direction.",
        "",
        "## Counts",
        "",
        "| Dataset | Total | Good | Acceptable | Failed | Mean IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in sorted({r.dataset for r in records}):
        ds = [r for r in records if r.dataset == dataset]
        good = sum(r.status == "good" for r in ds)
        acceptable = sum(r.status == "acceptable" for r in ds)
        failed = sum(r.status.startswith("failed") for r in ds)
        mean_iou = sum(r.iou for r in ds) / len(ds) if ds else 0.0
        lines.append(
            f"| {dataset} | {len(ds)} | {good} | {acceptable} | {failed} | {mean_iou:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Selected Examples",
            "",
        ]
    )
    for record in selected:
        rel = Path(record.panel_path).relative_to(out_root)
        lines.extend(
            [
                f"### {record.dataset} / {record.stem} / {record.status}",
                "",
                f"- IoU: {record.iou:.3f}",
                f"- predicted mask area ratio: {record.pred_area_ratio:.4f}",
                f"- panel: `{rel}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation For Report",
            "",
            "- ROI extraction is feasible when the predicted mask overlaps the wound region.",
            "- Some examples fail or produce low-quality ROIs, so classification performance",
            "  will depend strongly on segmentation quality.",
            "- This supports keeping the classification direction as a feasibility track",
            "  while building a cleaner ulcer-boundary baseline in parallel.",
            "",
        ]
    )
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/segmentation", type=Path)
    parser.add_argument(
        "--pred-root",
        default="outputs/predictions/report_2026_06_22/roi_feasibility_v3",
        type=Path,
    )
    parser.add_argument("--out-root", default="outputs/reports/2026-06-22_roi_demo", type=Path)
    parser.add_argument("--datasets", nargs="+", default=["abrasion", "cut", "laceration"])
    parser.add_argument("--margin-ratio", type=float, default=0.15)
    parser.add_argument("--selected-per-status", type=int, default=2)
    args = parser.parse_args()

    all_records: list[RoiRecord] = []
    for dataset in args.datasets:
        all_records.extend(
            process_dataset(
                dataset=dataset,
                raw_root=args.raw_root,
                pred_root=args.pred_root,
                out_root=args.out_root,
                margin_ratio=args.margin_ratio,
            )
        )

    selected = choose_selected(all_records, args.selected_per_status)
    copy_selected(selected, args.out_root)
    write_csv(all_records, args.out_root)
    write_summary(all_records, selected, args.out_root)
    print(f"ROI demo records: {len(all_records)}")
    print(f"Selected panels: {len(selected)}")
    print(f"Output: {args.out_root}")


if __name__ == "__main__":
    main()
