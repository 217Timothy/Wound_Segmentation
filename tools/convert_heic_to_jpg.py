#!/usr/bin/env python3
"""Convert HEIC/PNG/BMP/TIFF images in a folder to JPG and remove the originals.

Example:
    python tools/convert_heic_to_jpg.py --dir data/raw/segmentation/TKR/images
"""

import argparse
import os

from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

register_heif_opener()

_CONVERTIBLE = (".heic", ".png", ".jpeg", ".bmp", ".tiff")


def convert_folder(directory: str, quality: int = 95) -> None:
    files = [f for f in sorted(os.listdir(directory))
             if f.lower().endswith(_CONVERTIBLE)]
    if not files:
        print(f"[convert] nothing to convert in {directory}")
        return

    print(f"[convert] converting {len(files)} files in {directory}")
    for fname in tqdm(files, desc="convert"):
        src = os.path.join(directory, fname)
        dst = os.path.join(directory, f"{os.path.splitext(fname)[0]}.jpg")
        try:
            Image.open(src).convert("RGB").save(dst, "JPEG", quality=quality)
            if os.path.exists(dst) and os.path.abspath(src) != os.path.abspath(dst):
                os.remove(src)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[convert][error] {fname}: {exc}")
    print("[convert] done")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Folder of images to convert.")
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    convert_folder(args.dir, args.quality)


if __name__ == "__main__":
    main()
