"""Convert new chronic-wound annotations into segmentation-style masks.

Outputs are written in the project convention:

    data/raw/segmentation/<name>/images
    data/raw/segmentation/<name>/masks

Default outputs:

- ``footulcer_tissue``: DFUTissue class-ID tissue masks.
- ``chronic_bbox``: wound_localization bbox annotations converted to 0/255
  rectangle masks.

The wound_localization dataset ships the same bbox annotations in VOC, YOLO,
and COCO formats. The default uses VOC, but ``--bbox-format`` can switch the
source annotation format when needed.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class ConversionSummary:
    dataset: str
    annotation_format: str
    source: str
    destination: str
    images: int = 0
    masks: int = 0
    annotation_items: int = 0
    skipped: int = 0
    note: str = ""


def natural_key(path: Path) -> tuple[int, str]:
    if path.stem.isdigit():
        return (int(path.stem), path.stem)
    return (10**9, path.stem)


def normalized_stem(stem: str) -> str:
    if stem.isdigit():
        return f"{int(stem):04d}"
    return stem.lower().replace(" ", "_")


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_by_stem(folder: Path, stem: str) -> Path | None:
    for ext in sorted(IMAGE_EXTS):
        candidate = folder / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask):
        raise RuntimeError(f"Failed to write mask: {path}")


def clamp_box(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(width - 1, int(math.floor(xmin))))
    y1 = max(0, min(height - 1, int(math.floor(ymin))))
    x2 = max(0, min(width - 1, int(math.ceil(xmax))))
    y2 = max(0, min(height - 1, int(math.ceil(ymax))))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def boxes_to_mask(
    boxes: list[tuple[int, int, int, int]],
    width: int,
    height: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        mask[ymin : ymax + 1, xmin : xmax + 1] = 255
    return mask


def parse_voc_boxes(xml_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    root = ET.parse(xml_path).getroot()
    boxes: list[tuple[int, int, int, int]] = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        values: dict[str, float] = {}
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            node = bndbox.find(tag)
            values[tag] = float(node.text) if node is not None and node.text else 0.0
        box = clamp_box(
            values["xmin"],
            values["ymin"],
            values["xmax"],
            values["ymax"],
            width,
            height,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def parse_yolo_boxes(txt_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, x_center, y_center, box_width, box_height = parts[:5]
        xc = float(x_center) * width
        yc = float(y_center) * height
        bw = float(box_width) * width
        bh = float(box_height) * height
        box = clamp_box(
            xc - bw / 2,
            yc - bh / 2,
            xc + bw / 2,
            yc + bh / 2,
            width,
            height,
        )
        if box is not None:
            boxes.append(box)
    return boxes


def load_coco_boxes(coco_path: Path) -> dict[str, list[tuple[float, float, float, float]]]:
    data = json.loads(coco_path.read_text(encoding="utf-8"))
    image_by_id = {item["id"]: item["file_name"] for item in data.get("images", [])}
    boxes_by_file: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for annotation in data.get("annotations", []):
        file_name = image_by_id.get(annotation.get("image_id"))
        bbox = annotation.get("bbox")
        if file_name is None or bbox is None or len(bbox) != 4:
            continue
        x, y, width, height = [float(value) for value in bbox]
        boxes_by_file[file_name].append((x, y, x + width, y + height))
    return boxes_by_file


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        (path for path in folder.iterdir() if path.suffix.lower() in IMAGE_EXTS),
        key=natural_key,
    )


def convert_dfutissue(
    chronic_root: Path,
    raw_root: Path,
    output_name: str,
) -> ConversionSummary:
    source_root = chronic_root / "DFUTissueSegNet-main" / "DFUTissue" / "Labeled" / "Original"
    output_root = raw_root / "segmentation" / output_name

    copied = 0
    skipped = 0
    non_empty = 0
    for split in ("TrainVal", "Test"):
        image_dir = source_root / "Images" / split
        annotation_dir = source_root / "Annotations" / split
        if not image_dir.exists() or not annotation_dir.exists():
            continue
        for image_path in iter_image_files(image_dir):
            annotation_path = find_by_stem(annotation_dir, image_path.stem)
            annotation = None
            if annotation_path is not None:
                annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
            if annotation_path is None or annotation is None:
                skipped += 1
                continue

            out_stem = f"dfutissue_{normalized_stem(image_path.stem)}"
            copy_file(image_path, output_root / "images" / f"{out_stem}{image_path.suffix.lower()}")
            save_mask(annotation, output_root / "masks" / f"{out_stem}.png")
            non_empty += int(np.count_nonzero(annotation) > 0)
            copied += 1

    palette = source_root / "Palette" / "palette_colorCode.txt"
    if palette.exists():
        copy_file(palette, output_root / "palette_colorCode.txt")

    return ConversionSummary(
        dataset="DFUTissue",
        annotation_format="class_id_mask",
        source=str(source_root),
        destination=str(output_root),
        images=copied,
        masks=copied,
        annotation_items=non_empty,
        skipped=skipped,
        note="Tissue labels are saved as class-ID masks, not binary boundary masks.",
    )


def get_localization_boxes(
    *,
    image_path: Path,
    annotation_root: Path,
    bbox_format: str,
    width: int,
    height: int,
    coco_boxes: dict[str, list[tuple[float, float, float, float]]] | None,
) -> list[tuple[int, int, int, int]]:
    if bbox_format == "voc":
        xml_path = annotation_root / "VOC_Format" / f"{image_path.stem}.xml"
        if not xml_path.exists():
            return []
        return parse_voc_boxes(xml_path, width, height)
    if bbox_format == "yolo":
        txt_path = annotation_root / "YOLO_Format" / f"{image_path.stem}.txt"
        if not txt_path.exists():
            return []
        return parse_yolo_boxes(txt_path, width, height)
    if bbox_format == "coco":
        boxes: list[tuple[int, int, int, int]] = []
        assert coco_boxes is not None
        for xmin, ymin, xmax, ymax in coco_boxes.get(image_path.name, []):
            box = clamp_box(xmin, ymin, xmax, ymax, width, height)
            if box is not None:
                boxes.append(box)
        return boxes
    raise ValueError(f"Unsupported bbox format: {bbox_format}")


def convert_localization(
    chronic_root: Path,
    raw_root: Path,
    output_name: str,
    bbox_format: str,
) -> ConversionSummary:
    source_root = chronic_root / "wound_localization-main" / "dataset" / "Localization_ROI_Dataset"
    image_root = source_root / "images"
    annotation_root = source_root / "annotations"
    output_root = raw_root / "segmentation" / output_name

    coco_boxes = None
    if bbox_format == "coco":
        coco_boxes = load_coco_boxes(annotation_root / "COCO_Format" / "wound.json")

    copied = 0
    skipped = 0
    object_count = 0
    for image_path in iter_image_files(image_root):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped += 1
            continue
        height, width = image.shape[:2]
        boxes = get_localization_boxes(
            image_path=image_path,
            annotation_root=annotation_root,
            bbox_format=bbox_format,
            width=width,
            height=height,
            coco_boxes=coco_boxes,
        )
        if not boxes:
            skipped += 1
            continue

        out_stem = f"localization_{normalized_stem(image_path.stem)}"
        copy_file(image_path, output_root / "images" / f"{out_stem}{image_path.suffix.lower()}")
        save_mask(boxes_to_mask(boxes, width, height), output_root / "masks" / f"{out_stem}.png")
        copied += 1
        object_count += len(boxes)

    return ConversionSummary(
        dataset="wound_localization",
        annotation_format=bbox_format,
        source=str(source_root),
        destination=str(output_root),
        images=copied,
        masks=copied,
        annotation_items=object_count,
        skipped=skipped,
        note="BBox annotations are converted to 0/255 rectangle masks.",
    )


def write_manifest(raw_root: Path, summaries: list[ConversionSummary]) -> None:
    manifest = {
        "generated_by": "tools/preprocess/convert_new_annotations_to_masks.py",
        "important_note": (
            "These outputs are segmentation-style masks, but footulcer_tissue is "
            "multi-class tissue labeling and chronic_bbox is weak bbox labeling. "
            "Do not mix them into default boundary training without an explicit experiment."
        ),
        "outputs": [asdict(summary) for summary in summaries],
    }
    path = raw_root / "segmentation" / "annotation_mask_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DFUTissue and wound_localization annotations to masks."
    )
    parser.add_argument("--chronic-root", default="ChronicWoundDataset", type=Path)
    parser.add_argument("--raw-root", default="data/raw", type=Path)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["dfutissue", "localization"],
        choices=["dfutissue", "localization"],
    )
    parser.add_argument("--dfutissue-name", default="footulcer_tissue")
    parser.add_argument("--localization-name", default="chronic_bbox")
    parser.add_argument("--bbox-format", default="voc", choices=["voc", "yolo", "coco"])
    args = parser.parse_args()

    summaries: list[ConversionSummary] = []
    if "dfutissue" in args.datasets:
        summaries.append(convert_dfutissue(args.chronic_root, args.raw_root, args.dfutissue_name))
    if "localization" in args.datasets:
        summaries.append(
            convert_localization(
                args.chronic_root,
                args.raw_root,
                args.localization_name,
                args.bbox_format,
            )
        )

    write_manifest(args.raw_root, summaries)
    for summary in summaries:
        print(
            f"{summary.dataset} -> {summary.destination}: "
            f"images={summary.images}, masks={summary.masks}, "
            f"annotation_items={summary.annotation_items}, skipped={summary.skipped}"
        )


if __name__ == "__main__":
    main()
