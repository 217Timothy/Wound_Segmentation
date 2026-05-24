#!/usr/bin/env python3
"""Rename every image in a folder to '<prefix>_<index>.png' (sorted order).

Example:
    python tools/rename_images.py --dir data/raw/segmentation/laceration --prefix laceration
"""

import argparse
import os

_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def rename_images(folder: str, prefix: str, start_idx: int = 1) -> None:
    files = [f for f in sorted(os.listdir(folder))
             if os.path.isfile(os.path.join(folder, f))
             and f.lower().endswith(_IMAGE_EXTS)]

    idx = start_idx
    for fname in files:
        new_name = f"{prefix}_{idx:03d}.png"
        os.rename(os.path.join(folder, fname), os.path.join(folder, new_name))
        print(f"{fname} -> {new_name}")
        idx += 1
    print(f"[rename] renamed {len(files)} files in {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Folder of images to rename.")
    parser.add_argument("--prefix", required=True, help="Filename prefix.")
    parser.add_argument("--start", type=int, default=1, help="Starting index.")
    args = parser.parse_args()
    rename_images(args.dir, args.prefix, args.start)


if __name__ == "__main__":
    main()
