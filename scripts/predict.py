import os
import sys
import argparse
import glob
import cv2
import numpy as np
import torch
import albumentations as A
import albumentations.pytorch as ToTensorV2
from tqdm import tqdm
import shutil

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models.unet import UNet
from src.engine import infer_one_image
from src.utils import load_checkpoint, tensor_to_numpy, make_overlay, make_combine


# ==========================================
# 1. 設定參數與參數解析器 (Configuration and ArgParse)
# ==========================================
IMAGE_SIZE = 512
RUN_NAME = "unet_v1"
DELETE = True


def get_args():
    parser = argparse.ArgumentParser(description="Inference on images using U-Net")
    
    # 輸入與輸出路徑
    parser.add_argument("--input", type=str, default="data/processed",
                        help="輸入圖片的資料夾路徑")
    parser.add_argument("--output", type=str, default="results",
                        help="輸出結果的根目錄")
    parser.add_argument("--checkpoint", type=str, default=f"checkpoints/{RUN_NAME}/best.pt",
                        help="模型權重檔路徑 (.pt)")
    
    # 其他設定
    parser.add_argument("--dataset", type=str, required=True,
                        help="資料集名稱")
    parser.add_argument("--split", type=str, default="test",
                        help="要預測哪個切分? (預設 test)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用設備 (cuda/cpu/mps)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="判定傷口的門檻值 (0.0 ~ 1.0)")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    input_dir = os.path.join(args.input, args.dataset, args.split, "images")
    # 1. 建立輸出資料夾結構
    pred_dir = os.path.join(args.output, "predictions", args.dataset)
    viz_dir = os.path.join(args.output, "visualizations")
    overlay_dir = os.path.join(viz_dir, "overlay", args.dataset)
    combine_dir = os.path.join(viz_dir, "combine", args.dataset)
    
    print(f"[INFO] Input Folder: {input_dir}")
    print(f"[INFO] Output Root Folder: {args.output}")
    print(f"[INFO] Predict Mask Output Folder: {pred_dir}")
    print(f"[INFO] Overlay Image Output Folder: {overlay_dir}")
    print(f"[INFO] Combine Image Output Folder: {combine_dir}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Threshold: {args.threshold}")
    
    if DELETE and os.path.exists(pred_dir) and os.path.exists(overlay_dir) and os.path.exists(combine_dir):
        shutil.rmtree(pred_dir)
        shutil.rmtree(overlay_dir)
        shutil.rmtree(combine_dir)
    
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(combine_dir, exist_ok=True)
    
    # 2. 載入模型
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        return
    
    print("[INFO] Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(args.device)
    load_checkpoint(args.checkpoint, model)
    
    # 3. 定義預處理
    # 推論時只做 Resize & Normalize
    transform = A.Compose([
        A.Resize(height=512, width=512),
    ])
    
    # 4. 抓圖片
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    img_paths = []
    for ext in extensions:
        img_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not img_paths:
        print(f"[Warn] No images found in {input_dir}")
        return
    
    # 5. 開始推論整個資料夾的test
    print(f"[INFO] Found {len(img_paths)} images. Processing...")
    for img_path in tqdm(img_paths):
        filename = os.path.basename(img_path)
        name_no_ext = os.path.splitext(filename)[0]
        
        # A. 讀圖
        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # B. 預處理 - Step 1: Resize (用 Albumentations)
        augmented = transform(image=img_rgb)
        img_resized = augmented["image"] # 這時候還是 Numpy (512, 512, 3), 0-255
        
        img_float = img_resized.astype(np.float32) / 255.0
        im_transpose = img_float.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
        img_tensor = torch.from_numpy(im_transpose)
        
        # C. 核心推論
        pred_mask = infer_one_image(
            model, 
            img_tensor, 
            args.device, 
            args.threshold
        )
        pred_mask = pred_mask.squeeze()
        
        # D. 儲存結果
        # a. 存 Predict Mask
        cv2.imwrite(os.path.join(pred_dir, f"{name_no_ext}.png"), pred_mask * 255)
        
        img_vis = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
        # b. 存 Visualization 的 Overlay
        overlay = make_overlay(img_vis, pred_mask)
        cv2.imwrite(os.path.join(overlay_dir, f"{name_no_ext}.png"), overlay)
        
        # c. 存 Visualization 的 Combine
        combine = make_combine(img_vis, pred_mask)
        cv2.imwrite(os.path.join(combine_dir, f"{name_no_ext}.png"), combine)
    print(f"✅ All done! Results: {args.output}")


if __name__ == "__main__":
    main()