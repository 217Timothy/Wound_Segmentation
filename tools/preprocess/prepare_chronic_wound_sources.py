"""Stage public chronic-wound sources into clearly named raw folders.

This script does not modify the original downloaded folders. It copies the
usable public sources from ``ChronicWoundDataset`` into clean, source-explicit
folders under ``data/raw`` so downstream preprocessing does not have to guess
what "DFU" means.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class SourceSummary:
    name: str
    task: str
    clinical_scope: str
    source: str
    destination: str
    images: int = 0
    masks: int = 0
    annotations: int = 0
    included_in_default_boundary_training: bool = False
    note: str = ""


def reset_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{path} already exists. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def find_mask(mask_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = mask_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def stage_pair_folder(
    *,
    source_name: str,
    clinical_scope: str,
    source_root: Path,
    output_root: Path,
    split_specs: list[tuple[str, Path, Path]],
    output_dataset: str,
    default_training: bool,
    note: str,
) -> SourceSummary:
    dst = output_root / "segmentation_clean" / output_dataset
    image_dst = dst / "images"
    mask_dst = dst / "masks"
    image_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing_masks = 0
    for split_name, img_dir, mask_dir in split_specs:
        if not img_dir.exists() or not mask_dir.exists():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            mask_path = find_mask(mask_dir, img_path.stem)
            if mask_path is None:
                missing_masks += 1
                continue
            new_stem = f"{source_name}_{split_name}_{img_path.stem}"
            copy_file(img_path, image_dst / f"{new_stem}{img_path.suffix.lower()}")
            copy_file(mask_path, mask_dst / f"{new_stem}{mask_path.suffix.lower()}")
            copied += 1

    return SourceSummary(
        name=output_dataset,
        task="binary_segmentation",
        clinical_scope=clinical_scope,
        source=str(source_root),
        destination=str(dst),
        images=copied,
        masks=copied,
        included_in_default_boundary_training=default_training,
        note=note + (f" Missing masks skipped: {missing_masks}." if missing_masks else ""),
    )


def stage_fuseg(chronic_root: Path, output_root: Path) -> SourceSummary:
    root = chronic_root / "wound-segmentation-master" / "data" / "Foot Ulcer Segmentation Challenge"
    return stage_pair_folder(
        source_name="fuseg",
        clinical_scope="foot_ulcer",
        source_root=root,
        output_root=output_root,
        split_specs=[
            ("train", root / "train" / "images", root / "train" / "labels"),
            ("validation", root / "validation" / "images", root / "validation" / "labels"),
        ],
        output_dataset="FUSeg_FootUlcer",
        default_training=True,
        note=(
            "FUSeg foot-ulcer boundary masks. Test images have no masks and are "
            "not staged for supervised segmentation."
        ),
    )


def stage_medetec(chronic_root: Path, output_root: Path) -> SourceSummary:
    root = chronic_root / "wound-segmentation-master" / "data" / "Medetec_foot_ulcer_224"
    return stage_pair_folder(
        source_name="medetec",
        clinical_scope="foot_ulcer",
        source_root=root,
        output_root=output_root,
        split_specs=[
            ("train", root / "train" / "images", root / "train" / "labels"),
            ("test", root / "test" / "images", root / "test" / "labels"),
        ],
        output_dataset="Medetec_FootUlcer",
        default_training=True,
        note="Small preprocessed foot-ulcer segmentation set.",
    )


def stage_azh_patches(chronic_root: Path, output_root: Path) -> SourceSummary:
    zip_path = (
        chronic_root
        / "wound-segmentation-master"
        / "data"
        / "wound_dataset"
        / "azh_wound_care_center_dataset_patches.zip"
    )
    dst = output_root / "segmentation_clean" / "AZH_WoundPatches"
    image_dst = dst / "images"
    mask_dst = dst / "masks"
    image_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        image_names = sorted(
            n
            for n in names
            if "/images/" in n and Path(n).suffix.lower() in IMAGE_EXTS
        )
        for image_name in image_names:
            rel = Path(image_name)
            split = rel.parts[0]
            stem = rel.stem
            label_name = f"{split}/labels/{rel.name}"
            if label_name not in names:
                continue
            new_name = f"azhpatch_{split}_{stem}.png"
            with zf.open(image_name) as src, open(image_dst / new_name, "wb") as dst_file:
                shutil.copyfileobj(src, dst_file)
            with zf.open(label_name) as src, open(mask_dst / new_name, "wb") as dst_file:
                shutil.copyfileobj(src, dst_file)
            copied += 1

    return SourceSummary(
        name="AZH_WoundPatches",
        task="binary_segmentation",
        clinical_scope="chronic_wound_patch_mixed_or_unknown",
        source=str(zip_path),
        destination=str(dst),
        images=copied,
        masks=copied,
        included_in_default_boundary_training=False,
        note=(
            "Patch-style AZH wound data. Kept separate because it is not full-image "
            "foot-ulcer data and should not be mixed into the clean ulcer baseline."
        ),
    )


def stage_dfutissue(chronic_root: Path, output_root: Path) -> SourceSummary:
    root = chronic_root / "DFUTissueSegNet-main" / "DFUTissue" / "Labeled" / "Original"
    dst = output_root / "tissue_clean" / "DFUTissue_DFU_TissueLabels"
    image_dst = dst / "images"
    ann_dst = dst / "annotations"
    image_dst.mkdir(parents=True, exist_ok=True)
    ann_dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for split in ("TrainVal", "Test"):
        img_dir = root / "Images" / split
        ann_dir = root / "Annotations" / split
        if not img_dir.exists() or not ann_dir.exists():
            continue
        for img_path in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            ann_path = find_mask(ann_dir, img_path.stem)
            if ann_path is None:
                continue
            new_stem = f"dfutissue_{split.lower()}_{img_path.stem}"
            copy_file(img_path, image_dst / f"{new_stem}{img_path.suffix.lower()}")
            copy_file(ann_path, ann_dst / f"{new_stem}{ann_path.suffix.lower()}")
            copied += 1

    palette = root / "Palette" / "palette_colorCode.txt"
    if palette.exists():
        copy_file(palette, dst / "palette_colorCode.txt")

    return SourceSummary(
        name="DFUTissue_DFU_TissueLabels",
        task="tissue_segmentation_reference",
        clinical_scope="dfu",
        source=str(root),
        destination=str(dst),
        images=copied,
        annotations=copied,
        included_in_default_boundary_training=False,
        note=(
            "DFU tissue labels. Kept as tissue reference, not merged into binary "
            "boundary training by default."
        ),
    )


def stage_localization(chronic_root: Path, output_root: Path) -> SourceSummary:
    root = chronic_root / "wound_localization-main" / "dataset" / "Localization_ROI_Dataset"
    dst = output_root / "detection_clean" / "AZH_ChronicUlcerMixed_BBox"
    for sub in ("images", "annotations/VOC_Format", "annotations/YOLO_Format", "annotations/COCO_Format"):
        (dst / sub).mkdir(parents=True, exist_ok=True)

    image_count = 0
    for img_path in sorted((root / "images").iterdir()):
        if img_path.suffix.lower() in IMAGE_EXTS:
            copy_file(img_path, dst / "images" / f"azh_bbox_{img_path.name}")
            image_count += 1

    ann_count = 0
    for ann_dir in ("VOC_Format", "YOLO_Format", "COCO_Format"):
        src_dir = root / "annotations" / ann_dir
        if not src_dir.exists():
            continue
        for ann_path in sorted(p for p in src_dir.iterdir() if p.is_file()):
            copy_file(ann_path, dst / "annotations" / ann_dir / ann_path.name)
            ann_count += 1

    return SourceSummary(
        name="AZH_ChronicUlcerMixed_BBox",
        task="detection_bbox",
        clinical_scope="mixed_chronic_ulcer_dfu_pu_vu",
        source=str(root),
        destination=str(dst),
        images=image_count,
        annotations=ann_count,
        included_in_default_boundary_training=False,
        note=(
            "README says this localization dataset mixes DFU, pressure ulcer, "
            "and venous ulcer. It has bounding boxes, not pixel masks."
        ),
    )


def write_manifest(output_root: Path, summaries: list[SourceSummary]) -> None:
    manifest = output_root / "clean_dataset_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps([asdict(s) for s in summaries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chronic-root", default="ChronicWoundDataset")
    parser.add_argument("--output-root", default="data/raw")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chronic_root = Path(args.chronic_root)
    output_root = Path(args.output_root)

    reset_dir(output_root / "segmentation_clean", args.overwrite)
    reset_dir(output_root / "tissue_clean", args.overwrite)
    reset_dir(output_root / "detection_clean", args.overwrite)

    summaries = [
        stage_fuseg(chronic_root, output_root),
        stage_medetec(chronic_root, output_root),
        stage_azh_patches(chronic_root, output_root),
        stage_dfutissue(chronic_root, output_root),
        stage_localization(chronic_root, output_root),
    ]
    write_manifest(output_root, summaries)

    print("Prepared clean raw sources:")
    for summary in summaries:
        print(
            f"- {summary.name}: images={summary.images}, masks={summary.masks}, "
            f"annotations={summary.annotations}, default_boundary="
            f"{summary.included_in_default_boundary_training}"
        )
    print(f"Manifest: {output_root / 'clean_dataset_manifest.json'}")


if __name__ == "__main__":
    main()
