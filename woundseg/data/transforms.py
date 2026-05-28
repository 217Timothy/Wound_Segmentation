"""Albumentations transform pipelines.

ImageNet normalization is used because the encoders are ImageNet-pretrained.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_to_tensor():
    return [A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD), ToTensorV2()]


def get_train_transforms(img_size=(512, 512)):
    """Default augmentation for general wound training."""
    w, h = img_size
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=35, p=0.5),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.2),
            A.ElasticTransform(alpha=1, sigma=50),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.2),
        A.Resize(height=h, width=w),
        *_normalize_to_tensor(),
    ])


def get_tkr_train_transforms(img_size=(512, 512)):
    """Gentler augmentation tuned for the TKR knee dataset."""
    w, h = img_size
    return A.Compose([
        A.Resize(height=h, width=w),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.05, rotate_limit=10,
                           border_mode=0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=5, p=0.2),
        *_normalize_to_tensor(),
    ])


def get_val_transforms(img_size=(512, 512)):
    """Validation / inference: resize + normalize only (no random augmentation)."""
    w, h = img_size
    return A.Compose([
        A.Resize(height=h, width=w),
        *_normalize_to_tensor(),
    ])


def build_train_transforms(augmentation: str, img_size=(512, 512)):
    """Pick a training augmentation pipeline by name ("default" or "tkr")."""
    if augmentation == "tkr":
        return get_tkr_train_transforms(img_size)
    if augmentation == "default":
        return get_train_transforms(img_size)
    raise ValueError(f"Unknown augmentation preset: {augmentation!r}")
