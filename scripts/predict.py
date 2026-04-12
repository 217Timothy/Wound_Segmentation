import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import UNet, ResUnet, EfficientUnet
from src.engine import infer_one_image
from src.utils import load_checkpoint, make_overlay, make_overlay_with_gt, make_combine
from src.datasets import get_val_transforms


# ==========================================
# Global
# ==========================================
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
IMAGE_SIZE = 512


# ==========================================
# Args
# ==========================================
def get_args():
    parser = argparse.ArgumentParser(description="Inference")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--version", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--in_root", type=str, default="data_raw")
    parser.add_argument("--out_root", type=str, default="results")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--delete", type=int, default=1)

    # postprocess
    parser.add_argument("--use_postprocess", type=int, default=1)
    parser.add_argument("--min_area", type=int, default=200)
    parser.add_argument("--closing_kernel", type=int, default=7)
    parser.add_argument("--dilate_iter", type=int, default=0)
    parser.add_argument("--blur_kernel", type=int, default=7)
    parser.add_argument("--blur_sigma", type=float, default=0)
    parser.add_argument("--open_kernel", type=int, default=0)

    args = parser.parse_args()

    # load yaml
    if args.config and os.path.exists(args.config):
        print(f"[INFO] Loading config: {args.config}")
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)

    return args


# ==========================================
# Model
# ==========================================
def build_model(version):
    print("[INFO] Initializing Model...")

    if version == "v1":
        return UNet(3, 1).to(DEVICE)

    elif version == "v2":
        return ResUnet(
            encoder_name="resnet50",
            encoder_weights=None,
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

    elif version == "v3":
        return EfficientUnet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

    elif version == "wound-finetune-v1":
        return EfficientUnet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

    else:
        raise ValueError("Unsupported version")


# ==========================================
# Collect images
# ==========================================
def collect_image_paths(input_dir):
    img_dir = os.path.join(input_dir, "images")

    if not os.path.exists(img_dir):
        raise ValueError(f"{img_dir} not found")

    extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.heic",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.HEIC"
    ]

    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(img_dir, ext)))

    return sorted(img_paths)


# ==========================================
# Find GT mask
# ==========================================
def find_gt_mask(input_dir, name_no_ext):
    mask_dir = os.path.join(input_dir, "masks")

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        path = os.path.join(mask_dir, name_no_ext + ext)
        if os.path.exists(path):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                return (m > 0).astype(np.uint8) * 255
    return None


# ==========================================
# Postprocess
# ==========================================
def postprocess_mask(mask, args):
    mask = (mask * 255).astype(np.uint8)

    if args.blur_kernel > 1:
        k = args.blur_kernel if args.blur_kernel % 2 else args.blur_kernel + 1
        mask = cv2.GaussianBlur(mask, (k, k), args.blur_sigma)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if args.closing_kernel > 0:
        k = args.closing_kernel if args.closing_kernel % 2 else args.closing_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if args.open_kernel > 0:
        k = args.open_kernel if args.open_kernel % 2 else args.open_kernel + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if args.dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=args.dilate_iter)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8) # type: ignore

    cleaned = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= args.min_area:
            cleaned[labels == i] = 255

    return (cleaned > 0).astype(np.uint8)


# ==========================================
# Main
# ==========================================
def main():
    args = get_args()

    ckpt = os.path.join("checkpoints", args.version, args.run_name, "best.pt")

    print(f"[INFO] Using checkpoint: {ckpt}")

    model = build_model(args.version)
    load_checkpoint(ckpt, model)
    model.eval()

    transform = get_val_transforms((IMAGE_SIZE, IMAGE_SIZE))

    for ds in args.datasets:
        print(f"\n[INFO] Dataset: {ds}")

        input_dir = os.path.join(args.in_root, ds)

        pred_dir = os.path.join(args.out_root, "predictions", args.version, args.run_name, ds)
        overlay_dir = os.path.join(args.out_root, "visualizations", args.version, args.run_name, "overlay", ds)
        combine_dir = os.path.join(args.out_root, "visualizations", args.version, args.run_name, "combine", ds)

        if args.delete:
            shutil.rmtree(pred_dir, ignore_errors=True)
            shutil.rmtree(overlay_dir, ignore_errors=True)
            shutil.rmtree(combine_dir, ignore_errors=True)

        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(combine_dir, exist_ok=True)

        img_paths = collect_image_paths(input_dir)

        print(f"[INFO] Found {len(img_paths)} images")

        for img_path in tqdm(img_paths):
            name = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # type: ignore
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

            gt = find_gt_mask(input_dir, name)

            tensor = transform(image=img_rgb)["image"]

            pred = infer_one_image(model, tensor, DEVICE, args.threshold).squeeze()

            if args.use_postprocess:
                pred = postprocess_mask(pred, args)

            cv2.imwrite(os.path.join(pred_dir, f"{name}.png"), pred * 255)

            img_vis = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            if gt is not None:
                overlay = make_overlay_with_gt(img_vis, pred, gt)
            else:
                overlay = make_overlay(img_vis, pred)

            combine = make_combine(img_vis, pred)

            cv2.imwrite(os.path.join(overlay_dir, f"{name}.png"), overlay)
            cv2.imwrite(os.path.join(combine_dir, f"{name}.png"), combine)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()