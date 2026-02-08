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
from src.utils import load_checkpoint, make_overlay, make_overlay_with_gt, make_combine


# ==========================================
# 設定參數與參數解析器 (Configuration and ArgParse)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512

def get_args():
    parser = argparse.ArgumentParser(description="Inference on images using U-Net")
    
    # 必要參數：模型版本與資料集
    parser.add_argument("--version", type=str, required=True,
                        help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="第幾次跑")
    parser.add_argument("--datasets", type=str, required=True, nargs='+',
                        help="資料集名稱")
    
    # 輸入與輸出根目錄
    parser.add_argument("--in_root", type=str, default="data/processed",
                        help="輸入圖片的資料夾路徑")
    parser.add_argument("--out_root", type=str, default="results",
                        help="輸出結果的根目錄")
    
    # 其他設定
    parser.add_argument("--split", type=str, default="test",
                        help="要預測哪個切分? (預設 test)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="判定傷口的門檻值 (0.0 ~ 1.0)")
    parser.add_argument("--delete", type=int, default=1,
                        help="要不要刪掉舊的推論")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")
    
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Threshold: {args.threshold}")
    
    
    # 1. 定義影像轉換/預處理
    # 推論時只做 Resize & Normalize
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    ])
    
    # 2. 載入模型
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    print("[INFO] Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    load_checkpoint(checkpoint_path, model)
    
    # 3. 資料準備與推論
    for ds in args.datasets:
        print(f"\n[INFO] Processing Dataset: {ds} ...")
        
        # A. 建立輸出資料夾結構
        input_dir = os.path.join(args.in_root, ds, args.split)
        pred_dir = os.path.join(args.out_root, "predictions", args.version, args.run_name, ds) # 最終輸出 dir
        viz_dir = os.path.join(args.out_root, "visualizations", args.version, args.run_name)
        overlay_dir = os.path.join(viz_dir, "overlay", ds) # 最終輸出 dir
        combine_dir = os.path.join(viz_dir, "combine", ds) # 最終輸出 dir
        print(f"[INFO] Input Folder: {input_dir}")
        print(f"[INFO] Predict Mask Output Folder: {pred_dir}")
        print(f"[INFO] Overlay Image Output Folder: {overlay_dir}")
        print(f"[INFO] Combine Image Output Folder: {combine_dir}")
        if args.delete and os.path.exists(pred_dir) and os.path.exists(overlay_dir) and os.path.exists(combine_dir):
            shutil.rmtree(pred_dir)
            shutil.rmtree(overlay_dir)
            shutil.rmtree(combine_dir)
        
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(combine_dir, exist_ok=True)
        
        # B. 抓圖片
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
        img_paths = []
        for ext in extensions:
            img_paths.extend(glob.glob(os.path.join(input_dir, "images", ext)))
        
        if not img_paths:
            print(f"[Warn] No images found in {input_dir}")
            continue
        
        # C. 開始推論整個資料夾的test
        print(f"[INFO] Found {len(img_paths)} images. Processing...")
        for img_path in tqdm(img_paths):
            filename = os.path.basename(img_path)
            name_no_ext = os.path.splitext(filename)[0]
            
            # a. 讀圖與轉換色彩空間
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            # b. 動態尋找對應的 GT Mask (解決 glob 順序可能不一致的問題)
            gt_mask = None
            # 嘗試常見的 mask 副檔名
            for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
                # 假設 mask 放在 ../mask/filename.ext
                possible_mask_path = os.path.join(input_dir, "masks", name_no_ext + ext)
                if os.path.exists(possible_mask_path):
                    gt_origin = cv2.imread(possible_mask_path, cv2.IMREAD_GRAYSCALE)
                    if gt_origin is not None:
                        # [重要修正] Resize GT Mask 讓它跟模型輸出一樣大 (512x512)
                        gt_mask = cv2.resize(gt_origin, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
                        # 確保是 0/1 二值化
                        gt_mask = (gt_mask > 0).astype(np.uint8) * 255
                    break # 找到就停止
            
            # b. 預處理 - Step 1: Resize (用 Albumentations)
            augmented = transform(image=img_rgb)
            img_resized = augmented["image"] # 這時候還是 Numpy (512, 512, 3), 0-255
            
            img_float = img_resized.astype(np.float32) / 255.0
            im_transpose = img_float.transpose((2, 0, 1)) # (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(im_transpose)
            
            # c. 核心推論
            pred_mask = infer_one_image(
                model, 
                img_tensor, 
                DEVICE, 
                args.threshold
            )
            pred_mask = pred_mask.squeeze()
            
            # d. 儲存結果
            # 存 Predict Mask
            cv2.imwrite(os.path.join(pred_dir, f"{name_no_ext}.png"), pred_mask * 255)
            
            img_vis = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            
            # 存 Visualization 的 Overlay
            if args.split == "test":
                overlay = make_overlay(img_vis, pred_mask)
                cv2.imwrite(os.path.join(overlay_dir, f"{name_no_ext}.png"), overlay)
            elif args.split == "val":
                overlay = make_overlay_with_gt(img_vis, pred_mask, gt_mask)
                cv2.imwrite(os.path.join(overlay_dir, f"{name_no_ext}.png"), overlay)
            
            # 存 Visualization 的 Combine
            combine = make_combine(img_vis, pred_mask)
            cv2.imwrite(os.path.join(combine_dir, f"{name_no_ext}.png"), combine)
    print(f"\n✅ All done! Results: {args.out_root}")


if __name__ == "__main__":
    main()