"""Visualization helpers for masks and predictions.

All functions work on OpenCV-style BGR uint8 images and binary masks.
Masks are normalized internally to uint8 arrays with values in {0, 255}.
"""

import cv2
import numpy as np
import torch

# BGR colors
_GREEN = (0, 255, 0)
_YELLOW = (0, 255, 255)
_BLUE = (255, 0, 0)
_RED = (0, 0, 255)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (C, H, W) float tensor in [0, 1] to an (H, W, C) uint8 image."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def _to_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize any mask to a uint8 array with values in {0, 255}."""
    mask = np.asarray(mask)
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def make_overlay(img: np.ndarray, mask: np.ndarray,
                 color: tuple = _GREEN, alpha: float = 0.5) -> np.ndarray:
    """Overlay a single mask on an image with a translucent fill and contour."""
    mask = _to_uint8_mask(mask)
    fill = np.zeros_like(img)
    fill[mask == 255] = color
    overlay = cv2.addWeighted(img, 1.0, fill, alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, _YELLOW, 2)
    return overlay


def make_overlay_with_gt(img: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray,
                         alpha: float = 0.5) -> np.ndarray:
    """Overlay prediction (blue contour) and ground truth (red contour) together."""
    pred_mask = _to_uint8_mask(pred_mask)
    gt_mask = _to_uint8_mask(gt_mask)

    fill = np.zeros_like(img)
    fill[pred_mask == 255] = _GREEN
    overlay = cv2.addWeighted(img, 1.0, fill, alpha, 0)

    pred_contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_contours, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, pred_contours, -1, _BLUE, 2)
    cv2.drawContours(overlay, gt_contours, -1, _RED, 2)
    return overlay


def make_combine(img: np.ndarray, pred: np.ndarray,
                 gt: np.ndarray | None = None) -> np.ndarray:
    """Horizontally stack [image | prediction] or [image | prediction | gt]."""
    pred_vis = cv2.cvtColor(_to_uint8_mask(pred), cv2.COLOR_GRAY2BGR)
    if gt is not None:
        gt_vis = cv2.cvtColor(_to_uint8_mask(gt), cv2.COLOR_GRAY2BGR)
        return np.hstack((img, pred_vis, gt_vis))
    return np.hstack((img, pred_vis))
