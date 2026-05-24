#!/usr/bin/env python3
"""Sub-sample the training split of selected datasets for fine-tuning.

Useful for class balancing: cap large classes to N samples while keeping
small classes intact. Overwrites the dataset's train.txt files in place.

Edit TARGET_NUM below, then run:
    python tools/make_finetune_split.py
"""

import os
import random

random.seed(42)

# Splits live under <data_root>/splits/<dataset>/train.txt
SPLIT_ROOT = "data/processed/wound/splits"

# dataset -> number of training samples to keep (None = keep all)
TARGET_NUM = {
    "DFU": 50,
    "Chronic": 50,
    "Abrasion": None,
    "Cut": None,
    "Laceration": None,
}


def reduce_split(dataset_name: str, keep: int | None) -> None:
    path = os.path.join(SPLIT_ROOT, dataset_name, "train.txt")
    if not os.path.exists(path):
        print(f"[split][warn] not found: {path}")
        return

    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    if keep is None or len(lines) <= keep:
        print(f"[split] {dataset_name}: keep all ({len(lines)})")
        return

    selected = random.sample(lines, keep)
    with open(path, "w") as f:
        f.write("\n".join(selected) + "\n")
    print(f"[split] {dataset_name}: {len(lines)} -> {keep}")


def main():
    for dataset_name, keep in TARGET_NUM.items():
        reduce_split(dataset_name, keep)


if __name__ == "__main__":
    main()
