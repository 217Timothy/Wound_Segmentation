import os
import sys
import argparse
import torch
import torch.nn as nn
import cv2
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import ResUnet, EfficientUnet
from src.models.unet import UNet
from src.datasets.transforms import get_val_transforms
from src.engine import infer_one_image
from src.utils import load_checkpoint, make_overlay, make_combine
from data_preprocess.preprocess_split import letterbox_resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True, help="第幾次跑")
    parser.add_argument("--threshold", type=float, default=0.5, help="二值化 threshold")
    return parser.parse_args()


def build_model(version, run_name):
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
            model.model.segmentation_head = nn.Sequential( # type: ignore
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
    dilation_iter=0
):
    """
    pred_mask: numpy array, shape (H, W), binary mask (0/1 or 0/255)
    return: cleaned binary mask, uint8, values 0/1
    """
    # 保證先轉成 0/255 的 uint8
    mask = (pred_mask > 0).astype(np.uint8) * 255

    # 1. Closing: 補小洞、接小斷裂
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 2. 可選：輕微膨脹，讓帶狀區域更連續
    if dilation_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilation_iter)

    # 3. Connected Components：移除小碎片
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)

    if num_labels <= 1:
        return (mask > 0).astype(np.uint8)

    # 找所有面積夠大的 component
    valid_components = []
    for i in range(1, num_labels):  # 0 是背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            valid_components.append((i, area))

    if len(valid_components) == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    # 4. 只保留最大 component
    largest_label = max(valid_components, key=lambda x: x[1])[0]
    cleaned[labels == largest_label] = 255

    return (cleaned > 0).astype(np.uint8)


def main():
    args = get_args()

    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")

    user_input_img_dir = input("Please enter the image dir (ex. tkr, ncku): ").strip()
    base_out_dir = f"result_single/{user_input_img_dir}"
    pred_dir = os.path.join(base_out_dir, "predictions")
    viz_dir = os.path.join(base_out_dir, "visualizations")
    overlay_dir = os.path.join(viz_dir, "overlay")
    combine_dir = os.path.join(viz_dir, "combine")

    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(combine_dir, exist_ok=True)

    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Threshold: {args.threshold}")

    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return

    print("[INFO] Loading model...")
    model = build_model(args.version, args.run_name)
    load_checkpoint(checkpoint_path, model)
    model.eval()

    transform = get_val_transforms(img_size=(IMAGE_SIZE, IMAGE_SIZE))

    raw_input = input("Please enter the image name(s) (ex. 001 or 001 002 003): ").strip()
    img_list = [f"{user_input_img_dir}_{name}.jpg" for name in raw_input.split()]

    for img_name in img_list:
        img_path = os.path.join("data_raw", "test", user_input_img_dir, img_name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Image not found: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 先做和 preprocess 一樣的 letterbox
        img_lb = letterbox_resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

        # 再做和 validation 一樣的 normalize + tensor
        augmented = transform(image=img_lb)
        img_tensor = augmented["image"]

        pred_mask = infer_one_image(
            model=model,
            img_tensor=img_tensor,
            device=DEVICE,
            threshold=args.threshold
        )
        pred_mask = pred_mask.squeeze()
        pred_mask = postprocess_mask(
            pred_mask,
            min_area=200,
            closing_kernel_size=7,
            dilation_iter=0
        )

        # save mask
        name_no_ext = os.path.splitext(img_name)[0]
        cv2.imwrite(os.path.join(pred_dir, f"{name_no_ext}.png"), pred_mask * 255)

        # visualization
        img_vis = cv2.cvtColor(img_lb, cv2.COLOR_RGB2BGR)

        overlay = make_overlay(img_vis, pred_mask)
        cv2.imwrite(os.path.join(overlay_dir, f"{name_no_ext}.png"), overlay)

        combine = make_combine(img_vis, pred_mask)
        cv2.imwrite(os.path.join(combine_dir, f"{name_no_ext}.png"), combine)

        print(f"✅ {name_no_ext} done! Results: {base_out_dir}")

    print(f"\n✅ All done! Results: {base_out_dir}")


if __name__ == "__main__":
    main()