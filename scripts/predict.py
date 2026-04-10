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
# 全域設定
# ==========================================
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
IMAGE_SIZE = 512


def get_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None, help="Path to config file")
    known_args, remaining_args = conf_parser.parse_known_args()

    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        print(f"[INFO] Loading defaults from config: {known_args.config}")
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        parents=[conf_parser],
        description="Inference for segmentation model"
    )

    parser.add_argument("--version", type=str, help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, help="第幾次跑")
    parser.add_argument("--datasets", type=str, nargs="+", help="資料集名稱")

    parser.add_argument("--in_root", type=str, default="data", help="輸入圖片的根目錄")
    parser.add_argument("--out_root", type=str, default="results", help="輸出結果的根目錄")
    parser.add_argument("--split", type=str, default="val", help="要預測哪個切分")
    parser.add_argument("--input_mode", type=str, default="standard", help="standard / flat")
    parser.add_argument("--threshold", type=float, default=0.5, help="判定傷口的門檻值 (0.0 ~ 1.0)")
    parser.add_argument("--delete", type=int, default=1, help="要不要刪掉舊的推論")

    # postprocess
    parser.add_argument("--use_postprocess", type=int, default=1, help="是否啟用後處理")
    parser.add_argument("--min_area", type=int, default=200, help="最小保留面積")
    parser.add_argument("--closing_kernel", type=int, default=7, help="closing kernel size")
    parser.add_argument("--dilate_iter", type=int, default=0, help="dilation 次數")
    parser.add_argument("--blur_kernel", type=int, default=7, help="Gaussian blur kernel size")
    parser.add_argument("--blur_sigma", type=float, default=0, help="Gaussian blur sigma")
    parser.add_argument("--open_kernel", type=int, default=0, help="opening kernel size, 0 means disable")

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    return args


def build_model(version: str, run_name: str):
    print("[INFO] Initializing Model...")

    if version == "v1":
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)

    elif version == "v2":
        if "run2" in run_name:
            print("[INFO] Detecting Run 2 configuration: ResNet34 (No Attention)")
            encoder = "resnet34"
            attn_type = None

        elif "run3" in run_name:
            print("[INFO] Detecting Run 3 configuration: ResNet34 + scSE")
            encoder = "resnet34"
            attn_type = "scse"

        else:
            print("[INFO] Detecting configuration: ResNet50 + scSE + Dropout")
            encoder = "resnet50"
            attn_type = "scse"

        model = ResUnet(
            encoder_name=encoder,
            encoder_weights=None,
            decoder_attention_type=attn_type,
            classes=1
        ).to(DEVICE)

        if "run5" in run_name or "run6" in run_name:
            old_head = model.model.segmentation_head
            model.model.segmentation_head = nn.Sequential(  # type: ignore
                nn.Dropout2d(p=0.3),
                old_head
            )

    elif version == "v3":
        model = EfficientUnet(
            encoder_name="efficientnet-b3",
            encoder_weights=None,
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

    else:
        raise ValueError(f"Unsupported version: {version}")

    return model


def postprocess_mask(
    pred_mask,
    min_area=200,
    closing_kernel_size=7,
    dilation_iter=0,
    blur_kernel_size=7,
    blur_sigma=0,
    open_kernel_size=0
):
    """
    pred_mask: numpy array, shape (H, W), binary or soft mask
    return: cleaned binary mask, uint8, values 0/1
    """

    if pred_mask.dtype != np.uint8:
        pred_mask = pred_mask.astype(np.float32)

    if pred_mask.max() <= 1:
        mask = (pred_mask * 255).astype(np.uint8)
    else:
        mask = pred_mask.astype(np.uint8)

    if blur_kernel_size > 1:
        if blur_kernel_size % 2 == 0:
            blur_kernel_size += 1
        mask = cv2.GaussianBlur(mask, (blur_kernel_size, blur_kernel_size), blur_sigma)

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    if closing_kernel_size > 0:
        if closing_kernel_size % 2 == 0:
            closing_kernel_size += 1
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if open_kernel_size > 0:
        if open_kernel_size % 2 == 0:
            open_kernel_size += 1
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    if dilation_iter > 0:
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, dilate_kernel, iterations=dilation_iter)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    if num_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8)

    valid_components = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append((i, area))

    if len(valid_components) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    largest_label = max(valid_components, key=lambda x: x[1])[0]
    cleaned[labels == largest_label] = 255

    return (cleaned > 0).astype(np.uint8)


def collect_image_paths(input_dir: str, input_mode: str):
    extensions = [
        "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.heic",
        "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.HEIC"
    ]

    img_paths = []

    if input_mode == "flat":
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    else:
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(input_dir, "images", ext)))

    img_paths = sorted(img_paths)
    return img_paths


def find_gt_mask(input_dir: str, input_mode: str, name_no_ext: str):
    """
    只有 standard mode 才找 masks/
    flat mode 一律回傳 None
    """
    if input_mode != "standard":
        return None

    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        possible_mask_path = os.path.join(input_dir, "masks", name_no_ext + ext)
        if os.path.exists(possible_mask_path):
            gt_origin = cv2.imread(possible_mask_path, cv2.IMREAD_GRAYSCALE)
            if gt_origin is not None:
                gt_mask = cv2.resize(
                    gt_origin,
                    (IMAGE_SIZE, IMAGE_SIZE),
                    interpolation=cv2.INTER_NEAREST
                )
                gt_mask = (gt_mask > 0).astype(np.uint8) * 255
                return gt_mask
    return None


def main():
    args = get_args()

    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Datasets: {args.datasets}")
    print(f"[INFO] Input Mode: {args.input_mode}")
    print(f"[INFO] Split: {args.split}")
    print(f"[INFO] Threshold: {args.threshold}")
    print(f"[INFO] Use Postprocess: {bool(args.use_postprocess)}")
    print(f"[INFO] Postprocess min_area: {args.min_area}")
    print(f"[INFO] Postprocess closing_kernel: {args.closing_kernel}")
    print(f"[INFO] Postprocess dilate_iter: {args.dilate_iter}")
    print(f"[INFO] Postprocess blur_kernel: {args.blur_kernel}")
    print(f"[INFO] Postprocess blur_sigma: {args.blur_sigma}")
    print(f"[INFO] Postprocess open_kernel: {args.open_kernel}")

    transform = get_val_transforms(img_size=(IMAGE_SIZE, IMAGE_SIZE))

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    print("[INFO] Loading model...")
    model = build_model(args.version, args.run_name)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    for ds in args.datasets:
        print(f"\n[INFO] Processing Dataset: {ds} ...")

        if args.input_mode == "flat":
            input_dir = os.path.join(args.in_root, ds)
        else:
            input_dir = os.path.join(args.in_root, ds, args.split)

        pred_dir = os.path.join(args.out_root, "predictions", args.version, args.run_name, ds)
        viz_dir = os.path.join(args.out_root, "visualizations", args.version, args.run_name)
        overlay_dir = os.path.join(viz_dir, "overlay", ds)
        combine_dir = os.path.join(viz_dir, "combine", ds)

        print(f"[INFO] Input Folder: {input_dir}")
        print(f"[INFO] Predict Mask Output Folder: {pred_dir}")
        print(f"[INFO] Overlay Image Output Folder: {overlay_dir}")
        print(f"[INFO] Combine Image Output Folder: {combine_dir}")

        if not os.path.exists(input_dir):
            print(f"[Warn] Input folder not found: {input_dir}")
            continue

        if args.delete and os.path.exists(pred_dir):
            shutil.rmtree(pred_dir, ignore_errors=True)
        if args.delete and os.path.exists(overlay_dir):
            shutil.rmtree(overlay_dir, ignore_errors=True)
        if args.delete and os.path.exists(combine_dir):
            shutil.rmtree(combine_dir, ignore_errors=True)

        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(combine_dir, exist_ok=True)

        img_paths = collect_image_paths(input_dir, args.input_mode)

        if not img_paths:
            print(f"[Warn] No images found in {input_dir}")
            continue

        print(f"[INFO] Found {len(img_paths)} images. Processing...")

        for img_path in tqdm(img_paths):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[Warn] Failed to read image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)

            gt_mask = find_gt_mask(input_dir, args.input_mode, name_no_ext)

            augmented = transform(image=img_rgb)
            img_tensor = augmented["image"]

            pred_mask = infer_one_image(
                model,
                img_tensor,
                DEVICE,
                args.threshold
            )
            pred_mask = pred_mask.squeeze()

            if args.use_postprocess:
                pred_mask = postprocess_mask(
                    pred_mask,
                    min_area=args.min_area,
                    closing_kernel_size=args.closing_kernel,
                    dilation_iter=args.dilate_iter,
                    blur_kernel_size=args.blur_kernel,
                    blur_sigma=args.blur_sigma,
                    open_kernel_size=args.open_kernel
                )

            cv2.imwrite(os.path.join(pred_dir, f"{name_no_ext}.png"), pred_mask * 255)

            img_vis = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            if gt_mask is not None:
                overlay = make_overlay_with_gt(img_vis, pred_mask, gt_mask)
            else:
                overlay = make_overlay(img_vis, pred_mask)

            cv2.imwrite(os.path.join(overlay_dir, f"{name_no_ext}.png"), overlay)

            combine = make_combine(img_vis, pred_mask)
            cv2.imwrite(os.path.join(combine_dir, f"{name_no_ext}.png"), combine)

    print(f"\n✅ All done! Results: {args.out_root}")


if __name__ == "__main__":
    main()