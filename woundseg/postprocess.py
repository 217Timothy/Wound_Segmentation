"""Morphological post-processing for predicted masks.

Cleans up a raw binary prediction: optional blur, hole-closing, despeckling
and (optionally) keeping only the largest connected component — useful when
each image is expected to contain a single wound.
"""

import cv2
import numpy as np


def _odd(k: int) -> int:
    """Return the nearest odd kernel size >= 1."""
    k = max(int(k), 1)
    return k if k % 2 == 1 else k + 1


def postprocess_mask(mask: np.ndarray, *, blur_kernel: int = 7, blur_sigma: float = 0.0,
                     closing_kernel: int = 7, open_kernel: int = 0,
                     min_area: int = 200, keep_largest: bool = True) -> np.ndarray:
    """Clean a binary mask.

    Args:
        mask: (H, W) array with values in {0, 1} or {0, 255}.
        blur_kernel: Gaussian blur size before re-thresholding (<=1 disables).
        closing_kernel: morphological closing size (fills holes; 0 disables).
        open_kernel: morphological opening size (removes specks; 0 disables).
        min_area: drop connected components smaller than this many pixels.
        keep_largest: keep only the single largest component.

    Returns:
        (H, W) uint8 array with values in {0, 1}.
    """
    work = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)

    if blur_kernel > 1:
        k = _odd(blur_kernel)
        work = cv2.GaussianBlur(work, (k, k), blur_sigma)
    _, work = cv2.threshold(work, 127, 255, cv2.THRESH_BINARY)

    if closing_kernel > 0:
        k = _odd(closing_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)

    if open_kernel > 0:
        k = _odd(open_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(work, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(work)
    if contours:
        if keep_largest:
            contours = [max(contours, key=cv2.contourArea)]
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                cv2.drawContours(clean, [c], -1, 255, thickness=cv2.FILLED)

    if closing_kernel > 0:
        k = _odd(closing_kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return (clean > 0).astype(np.uint8)
