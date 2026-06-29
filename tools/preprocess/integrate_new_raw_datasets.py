"""Integrate newly downloaded chronic-wound datasets into raw folders.

The original downloads under ``ChronicWoundDataset`` are left untouched. This
script makes the new sources visible in the project's existing ``data/raw``
layout while keeping label types explicit:

- DFUTissue images are grouped as ``footulcer`` for multiclass use.
- DFUTissue tissue masks are staged as ``segmentation/footulcer_tissue``.
- wound_localization images are grouped as ``chronic`` for multiclass use.
- wound_localization VOC boxes are converted to weak rectangle masks under
  ``segmentation/chronic_bbox``.

The weak/tissue segmentation folders are intentionally not part of
``preprocess_wound.py``'s default binary-boundary training map.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class IntegrationSummary:
    name: str
    source: str
    multiclass_destination: str
    segmentation_destination: str
    images: int = 0
    masks: int = 0
    skipped: int = 0
    note: str = ""


def natural_key(path: Path) -> tuple[int, str]:
    if path.stem.isdigit():
        return (int(path.stem), path.stem)
    return (10**9, path.stem)


def normalized_numeric_stem(stem: str) -> str:
    if stem.isdigit():
        return f"{int(stem):04d}"
    return stem.lower().replace(" ", "_")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_matching_file(folder: Path, stem: str) -> Path | None:
    for ext in sorted(IMAGE_EXTS):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_dfutissue_pairs(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in ("TrainVal", "Test"):
        image_dir = root / "Images" / split
        annotation_dir = root / "Annotations" / split
        if not image_dir.exists() or not annotation_dir.exists():
            continue
        for image_path in sorted(
            (p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
            key=natural_key,
        ):
            annotation_path = find_matching_file(annotation_dir, image_path.stem)
            if annotation_path is not None:
                pairs.append((image_path, annotation_path))
    return pairs


def integrate_dfutissue(chronic_root: Path, raw_root: Path) -> IntegrationSummary:
    source_root = chronic_root / "DFUTissueSegNet-main" / "DFUTissue" / "Labeled" / "Original"
    multiclass_root = raw_root / "multiclass" / "footulcer"
    segmentation_root = raw_root / "segmentation" / "footulcer_tissue"

    copied = 0
    skipped = 0
    for image_path, annotation_path in iter_dfutissue_pairs(source_root):
        new_name = f"dfutissue_{normalized_numeric_stem(image_path.stem)}{image_path.suffix.lower()}"
        mask_name = f"{Path(new_name).stem}{annotation_path.suffix.lower()}"

        copy_file(image_path, multiclass_root / "images" / new_name)
        copy_file(annotation_path, multiclass_root / "masks" / mask_name)
        copy_file(image_path, segmentation_root / "images" / new_name)
        copy_file(annotation_path, segmentation_root / "masks" / mask_name)
        copied += 1

    palette_path = source_root / "Palette" / "palette_colorCode.txt"
    if palette_path.exists():
        copy_file(palette_path, segmentation_root / "palette_colorCode.txt")

    expected_images = sum(
        1
        for split in ("TrainVal", "Test")
        for p in (source_root / "Images" / split).glob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )
    skipped = max(0, expected_images - copied)
    return IntegrationSummary(
        name="DFUTissue",
        source=str(source_root),
        multiclass_destination=str(multiclass_root),
        segmentation_destination=str(segmentation_root),
        images=copied,
        masks=copied,
        skipped=skipped,
        note=(
            "DFU images grouped into multiclass/footulcer. Tissue annotations are "
            "kept as footulcer_tissue, not default binary wound-boundary masks."
        ),
    )


def parse_voc_boxes(xml_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    boxes: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        values = {}
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            node = box.find(tag)
            if node is None or node.text is None:
                values[tag] = 0.0
            else:
                values[tag] = float(node.text)

        xmin = max(0, min(width - 1, int(math.floor(values["xmin"]))))
        ymin = max(0, min(height - 1, int(math.floor(values["ymin"]))))
        xmax = max(0, min(width - 1, int(math.ceil(values["xmax"]))))
        ymax = max(0, min(height - 1, int(math.ceil(values["ymax"]))))
        if xmax >= xmin and ymax >= ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def iter_localization_images(root: Path) -> list[Path]:
    image_dir = root / "images"
    return sorted(
        (p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def integrate_localization(chronic_root: Path, raw_root: Path) -> IntegrationSummary:
    source_root = chronic_root / "wound_localization-main" / "dataset" / "Localization_ROI_Dataset"
    voc_root = source_root / "annotations" / "VOC_Format"
    multiclass_root = raw_root / "multiclass" / "chronic"
    segmentation_root = raw_root / "segmentation" / "chronic_bbox"

    copied = 0
    skipped = 0
    for image_path in iter_localization_images(source_root):
        normalized_stem = normalized_numeric_stem(image_path.stem)
        new_name = f"localization_{normalized_stem}{image_path.suffix.lower()}"
        copy_file(image_path, multiclass_root / new_name)

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        xml_path = voc_root / f"{image_path.stem}.xml"
        if image is None or not xml_path.exists():
            skipped += 1
            continue

        height, width = image.shape[:2]
        boxes = parse_voc_boxes(xml_path, width=width, height=height)
        if not boxes:
            skipped += 1
            continue

        mask = np.zeros((height, width), dtype=np.uint8)
        for xmin, ymin, xmax, ymax in boxes:
            mask[ymin : ymax + 1, xmin : xmax + 1] = 255

        copy_file(image_path, segmentation_root / "images" / new_name)
        mask_path = segmentation_root / "masks" / f"localization_{normalized_stem}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_path), mask)
        copied += 1

    return IntegrationSummary(
        name="wound_localization",
        source=str(source_root),
        multiclass_destination=str(multiclass_root),
        segmentation_destination=str(segmentation_root),
        images=copied,
        masks=copied,
        skipped=skipped,
        note=(
            "Mixed chronic-ulcer images grouped into multiclass/chronic. VOC boxes "
            "were converted to rectangle masks under chronic_bbox as weak labels."
        ),
    )


def write_manifest(raw_root: Path, summaries: list[IntegrationSummary]) -> None:
    manifest = {
        "generated_by": "tools/preprocess/integrate_new_raw_datasets.py",
        "important_note": (
            "footulcer_tissue and chronic_bbox are staged for analysis or weak-label "
            "experiments. Do not include them in default binary wound-boundary "
            "training unless the experiment explicitly expects tissue labels or bbox masks."
        ),
        "sources": [asdict(summary) for summary in summaries],
    }
    manifest_path = raw_root / "new_dataset_integration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Integrate DFUTissue and wound_localization into data/raw."
    )
    parser.add_argument("--chronic-root", default="ChronicWoundDataset", type=Path)
    parser.add_argument("--raw-root", default="data/raw", type=Path)
    args = parser.parse_args()

    summaries = [
        integrate_dfutissue(args.chronic_root, args.raw_root),
        integrate_localization(args.chronic_root, args.raw_root),
    ]
    write_manifest(args.raw_root, summaries)

    for summary in summaries:
        print(
            f"{summary.name}: images={summary.images}, masks={summary.masks}, "
            f"skipped={summary.skipped}"
        )


if __name__ == "__main__":
    main()
