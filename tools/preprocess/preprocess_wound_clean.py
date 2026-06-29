"""Build a clean processed wound-boundary dataset from staged raw sources."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

GROUPS = {
    "footulcer": ["FUSeg_FootUlcer", "Medetec_FootUlcer"],
    "ChronicUlcerMixed_Patches": ["AZH_WoundPatches"],
}


@dataclass
class Sample:
    image_path: Path
    mask_path: Path
    source_dataset: str


@dataclass
class GroupSummary:
    group: str
    sources: list[str]
    collected: int
    kept: int
    skipped_empty_or_tiny_mask: int
    skipped_huge_mask: int
    skipped_unreadable: int
    train: int
    val: int


def letterbox_resize(img: np.ndarray, size: tuple[int, int], is_mask: bool = False) -> np.ndarray:
    out_w, out_h = size
    img_h, img_w = img.shape[:2]
    scale = min(out_w / img_w, out_h / img_h)
    new_w = max(1, int(round(img_w * scale)))
    new_h = max(1, int(round(img_h * scale)))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

    if img.ndim == 3:
        canvas = np.zeros((out_h, out_w, img.shape[2]), dtype=np.uint8)
        canvas[(out_h - new_h) // 2 : (out_h - new_h) // 2 + new_h,
               (out_w - new_w) // 2 : (out_w - new_w) // 2 + new_w, :] = resized
    else:
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        canvas[(out_h - new_h) // 2 : (out_h - new_h) // 2 + new_h,
               (out_w - new_w) // 2 : (out_w - new_w) // 2 + new_w] = resized
    return canvas


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists. Pass --overwrite to rebuild it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def collect_samples(raw_root: Path, source_names: list[str]) -> list[Sample]:
    samples: list[Sample] = []
    for source_name in source_names:
        image_dir = raw_root / source_name / "images"
        mask_dir = raw_root / source_name / "masks"
        if not image_dir.exists() or not mask_dir.exists():
            print(f"[warn] missing source folder: {source_name}")
            continue
        for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            mask_path = find_mask(mask_dir, image_path.stem)
            if mask_path is None:
                continue
            samples.append(Sample(image_path=image_path, mask_path=mask_path, source_dataset=source_name))
    return samples


def read_pair(sample: Sample) -> tuple[np.ndarray | None, np.ndarray | None]:
    img = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        return None, None
    return img, mask


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    return np.where(mask >= 0.5, 255, 0).astype(np.uint8)


def save_pair(
    *,
    img: np.ndarray,
    mask: np.ndarray,
    out_dir: Path,
    name: str,
) -> None:
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "images" / name), img)
    cv2.imwrite(str(out_dir / "masks" / name), mask)


def build_group(
    *,
    group_name: str,
    sources: list[str],
    raw_root: Path,
    out_root: Path,
    image_size: int,
    val_ratio: float,
    seed: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
) -> GroupSummary:
    samples = collect_samples(raw_root, sources)
    kept: list[tuple[np.ndarray, np.ndarray, str]] = []
    rejected: list[dict[str, str | float]] = []
    skipped_unreadable = 0
    skipped_tiny = 0
    skipped_huge = 0

    for sample in tqdm(samples, desc=f"preprocess {group_name}"):
        img, mask = read_pair(sample)
        if img is None or mask is None:
            skipped_unreadable += 1
            rejected.append(
                {
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                    "source_dataset": sample.source_dataset,
                    "reason": "unreadable_image_or_mask",
                    "mask_area_ratio": "",
                }
            )
            continue
        mask = binarize_mask(mask)
        area_ratio = float(np.count_nonzero(mask)) / float(mask.size)
        if area_ratio < min_mask_ratio:
            skipped_tiny += 1
            rejected.append(
                {
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                    "source_dataset": sample.source_dataset,
                    "reason": "empty_or_tiny_mask",
                    "mask_area_ratio": area_ratio,
                }
            )
            continue
        if area_ratio > max_mask_ratio:
            skipped_huge += 1
            rejected.append(
                {
                    "image_path": str(sample.image_path),
                    "mask_path": str(sample.mask_path),
                    "source_dataset": sample.source_dataset,
                    "reason": "huge_mask",
                    "mask_area_ratio": area_ratio,
                }
            )
            continue

        img = letterbox_resize(img, (image_size, image_size), is_mask=False)
        mask = letterbox_resize(mask, (image_size, image_size), is_mask=True)
        kept.append((img, mask, sample.source_dataset))

    rng = random.Random(seed)
    rng.shuffle(kept)
    train_items, val_items = train_test_split(kept, test_size=val_ratio, random_state=seed)

    split_dir = out_root / "splits" / group_name
    split_dir.mkdir(parents=True, exist_ok=True)
    train_names: list[str] = []
    val_names: list[str] = []

    for split, items, names in (
        ("train", train_items, train_names),
        ("val", val_items, val_names),
    ):
        for idx, (img, mask, source_dataset) in enumerate(items):
            source_prefix = source_dataset.lower().replace("_", "")
            name = f"{group_name.lower()}_{source_prefix}_{idx:05d}.png"
            save_pair(img=img, mask=mask, out_dir=out_root / group_name / split, name=name)
            names.append(name)

    (split_dir / "train.txt").write_text("\n".join(train_names), encoding="utf-8")
    (split_dir / "val.txt").write_text("\n".join(val_names), encoding="utf-8")

    reject_dir = out_root / "rejected"
    reject_dir.mkdir(parents=True, exist_ok=True)
    reject_path = reject_dir / f"{group_name}.csv"
    with open(reject_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_path", "mask_path", "source_dataset", "reason", "mask_area_ratio"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rejected)

    return GroupSummary(
        group=group_name,
        sources=sources,
        collected=len(samples),
        kept=len(kept),
        skipped_empty_or_tiny_mask=skipped_tiny,
        skipped_huge_mask=skipped_huge,
        skipped_unreadable=skipped_unreadable,
        train=len(train_names),
        val=len(val_names),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw/segmentation_clean")
    parser.add_argument("--out-root", default="data/processed/wound_clean")
    parser.add_argument("--groups", nargs="+", default=["footulcer"], choices=sorted(GROUPS))
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-mask-ratio", type=float, default=0.0005)
    parser.add_argument("--max-mask-ratio", type=float, default=0.95)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    reset_dir(out_root, args.overwrite)

    summaries = []
    for group in args.groups:
        summaries.append(
            build_group(
                group_name=group,
                sources=GROUPS[group],
                raw_root=raw_root,
                out_root=out_root,
                image_size=args.image_size,
                val_ratio=args.val_ratio,
                seed=args.seed,
                min_mask_ratio=args.min_mask_ratio,
                max_mask_ratio=args.max_mask_ratio,
            )
        )

    summary_path = out_root / "preprocess_summary.json"
    summary_path.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Clean processed dataset summary:")
    for summary in summaries:
        print(
            f"- {summary.group}: collected={summary.collected}, kept={summary.kept}, "
            f"train={summary.train}, val={summary.val}, skipped_tiny="
            f"{summary.skipped_empty_or_tiny_mask}, skipped_huge={summary.skipped_huge_mask}, "
            f"skipped_unreadable={summary.skipped_unreadable}"
        )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
