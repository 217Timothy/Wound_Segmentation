import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
import albumentations as A
import albumentations.pytorch as ToTensorV2



current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import ResUnet, EfficientUnet
from src.datasets.transforms import get_val_transforms
from src.models.unet import UNet
from src.engine import infer_one_image
from src.utils import load_checkpoint, make_overlay, make_combine
from data_preprocess.preprocess_split import letterbox_resize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--version", type=str, required=True,
                        help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="第幾次跑")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")
    base_out_dir = "result_single"
    pred_dir = os.path.join(base_out_dir, "predictions")
    viz_dir = os.path.join(base_out_dir, "visualizations")
    overlay_dir = os.path.join(viz_dir, "overlay")
    combine_dir = os.path.join(viz_dir, "combine")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(combine_dir, exist_ok=True)
    
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    
    # 1. 定義影像轉換/預處理
    # 推論時只做 Resize & Normalize
    transform = get_val_transforms(img_size=(IMAGE_SIZE, IMAGE_SIZE))
    
    # 2. 載入模型
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    print("[INFO] Loading model...")
    print("[INFO] Initializing Model...")
    
    if args.version == "v1":
        model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    
    elif args.version == "v2":
        # 情境 a: Run 2 (ResNet34 + None)
        if "run2" in args.run_name:
            print("[INFO] Detecting Run 2 (or other) configuration: ResNet34 (No Attention)")
            encoder = "resnet34"
            attn_type = None
            
        # 情境 B: Run 3 (ResNet34 + scSE)
        elif "run3" in args.run_name:
            print("[INFO] Detecting Run 3 configuration: ResNet34 + scSE")
            encoder = "resnet34"
            attn_type = "scse"
        #情境 D: Run 4, 5, 6 (ResNet50 + scSE + Dropout)
        else:
            print("[INFO] configuration: ResNet50 + scSE + Dropout(p=0.5)")
            encoder = "resnet50"
            attn_type = "scse"
        
        model = ResUnet(
            encoder_name=encoder, 
            encoder_weights=None,  # 推論時設為 None，因為我們會載入 checkpoint
            decoder_attention_type=attn_type,
            classes=1
        ).to(DEVICE)
        # 🔥 如果是 Run 5, 6，必須裝上 Dropout 才能對齊權重檔
        if "run5" or "run6" in args.run_name:
            old_head = model.model.segmentation_head
            model.model.segmentation_head = nn.Sequential( # type: ignore
                nn.Dropout2d(p=0.3),    # 隨機丟棄 30% 的特徵圖，強迫模型學習更魯棒的特徵
                old_head
            )
    
    elif args.version == "v3":
        encoder = "efficientnet-b3"
        attn_type = "scse"
        # if args.run_name == "run1":
        #     encoder = "efficientnet-b4"
        #     attn_type = "scse"
        model = EfficientUnet(
            encoder_name=encoder, 
            encoder_weights=None,  # 推論時設為 None，因為我們會載入 checkpoint
            decoder_attention_type=attn_type,
            classes=1
        ).to(DEVICE)
    
    else:
        print("[Error] Unsupported Version")
        return
    
    load_checkpoint(checkpoint_path, model)
    
    
    # 3. 照片預處理
    user_input = input("Please enter the image name(ex. ncku_001): ")
    img_list = user_input.split(" ")
    img_list = [name + ".jpg" for name in img_list]
    for img_name in img_list:
        img_path = os.path.join("data_raw", "test", "ncku", img_name)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: 
            print("image not found")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = letterbox_resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        img_float = img.astype(np.float32) / 255.0
        im_transpose = img_float.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(im_transpose)
        
        pred_mask = infer_one_image(model, img_tensor, DEVICE)
        pred_mask = pred_mask.squeeze()
        
        # 存 Predict Mask
        cv2.imwrite(os.path.join(pred_dir, f"{img_name}.png"), pred_mask * 255)
        
        img_vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 存 Visualization 的 Overlay
        overlay = make_overlay(img_vis, pred_mask)
        cv2.imwrite(os.path.join(overlay_dir, f"{img_name}.png"), overlay)
        
        # 存 Visualization 的 Combine
        combine = make_combine(img_vis, pred_mask)
        cv2.imwrite(os.path.join(combine_dir, f"{img_name}.png"), combine)
        print(f"\n✅ {img_name} done! Results: {base_out_dir}")
    print(f"\n✅ All done! Results: {base_out_dir}")


if __name__ == "__main__":
    main()