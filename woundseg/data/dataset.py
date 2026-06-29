"""Wound segmentation dataset.

Expected on-disk layout (``root_dir`` is e.g. ``data/processed/wound``)::

    root_dir/
        splits/<dataset>/<split>.txt     # one file name per line
        <dataset>/<split>/images/<file>
        <dataset>/<split>/masks/<file>

One dataset object can span several `datasets` at once (multi-class training).
Each item is returned together with its dataset name so per-class metrics can
be computed during validation.
"""

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class WoundSegmentationDataset(Dataset):
    """Image/mask pairs for one or more wound datasets."""

    def __init__(self, root_dir, datasets, split="train", transform=None,
                 cache_data=False, skip_empty_masks=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.cache_data = cache_data
        self.skip_empty_masks = skip_empty_masks
        self.skipped_empty_masks = 0

        self.samples = self._collect_samples(datasets, split)
        print(f"[dataset] {split}: {len(self.samples)} samples "
              f"-> {dict(Counter(s[2] for s in self.samples))}")
        if self.skipped_empty_masks:
            print(f"[dataset] {split}: skipped {self.skipped_empty_masks} "
                  "empty-mask sample(s)")

        self._cache = self._build_cache(split) if cache_data else None

    # -- setup -------------------------------------------------------------
    def _collect_samples(self, datasets, split):
        """Return a list of (image_path, mask_path, dataset_name) tuples."""
        samples = []
        for name in datasets:
            split_file = self.root_dir / "splits" / name / f"{split}.txt"
            if not split_file.exists():
                print(f"[dataset][warn] missing split file: {split_file}")
                continue

            base = self.root_dir / name / split
            for line in split_file.read_text().splitlines():
                fname = line.strip()
                if not fname:
                    continue
                img_path = base / "images" / fname
                mask_path = base / "masks" / fname
                if img_path.exists() and mask_path.exists():
                    if self.skip_empty_masks and not self._mask_has_foreground(mask_path):
                        self.skipped_empty_masks += 1
                        continue
                    samples.append((img_path, mask_path, name))
        return samples

    @staticmethod
    def _mask_has_foreground(mask_path):
        """Return True when a mask contains at least one positive pixel."""
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        return mask is not None and bool(np.any(mask > 0))

    def _build_cache(self, split):
        """Pre-load every image/mask into RAM; drop unreadable files."""
        cache, valid = [], []
        for img_path, mask_path, name in tqdm(self.samples, desc=f"caching {split}"):
            img, mask = self._read(img_path, mask_path)
            if img is None or mask is None:
                print(f"[dataset][warn] skipping unreadable pair: {img_path}")
                continue
            cache.append((img, mask))
            valid.append((img_path, mask_path, name))
        self.samples = valid
        return cache

    @staticmethod
    def _read(img_path, mask_path):
        """Read an image (RGB) and mask (grayscale); return (None, None) on failure."""
        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            return None, None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), mask

    # -- dataset protocol --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx][2]
        if self._cache is not None:
            img, mask = self._cache[idx]
            img, mask = img.copy(), mask.copy()
        else:
            img_path, mask_path, _ = self.samples[idx]
            img, mask = self._read(img_path, mask_path)
            if img is None:
                raise RuntimeError(f"Cannot read sample: {img_path}")

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # Binarize the mask and force shape (1, H, W) regardless of transform.
        if isinstance(mask, torch.Tensor):
            mask = (mask > 0).float()
        else:
            mask = torch.from_numpy((mask > 0).astype(np.float32))
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return img, mask, name
