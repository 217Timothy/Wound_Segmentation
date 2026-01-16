import os
import sys
import json
import argparse
import numpy as np
import torch
import albumentations as A
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models.unet import UNet
from src.datasets.wound_dataset import SegmentationDataset
from src.engine import infer_one_image as infer_one_image
from src.utils import load_checkpoint


# ==========================================
# 1. 設定參數與參數解析器
# ==========================================
IMAGE_SIZE = 512
RUN_NAME = "unet_v1" # 這裡可以根據需要改成參數輸入，目前寫死也可以


def get_args():
    parser = argparse.ArgumentParser(description="Inference on images using U-Net")
    
    # 必要參數
    parser.add_argument("--dataset", type=str, required=True,
                        help="資料集名稱 (例如 WoundSeg)")
    
    # 路徑設定
    parser.add_argument("--root", type=str, default="data/processed",
                        help="資料集根目錄")
    parser.add_argument("--split", type=str, default="val",
                        help="要評估的清單 (val)")
    parser.add_argument("--checkpoint", type=str, default=f"checkpoints/{RUN_NAME}/best.pt",
                        help="模型權重路徑")
    parser.add_argument("--output", type=str, default=f"results/metrics/metrics_{RUN_NAME}.json",
                        help="評估報告輸出路徑 (.json)")
    
    # 其他
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用設備")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化門檻")
    
    return parser.parse_args()


def calculate_dice(pred, target):
    """
    計算單張圖的 Dice Score
    Args:
        pred: (H, W) 0/1 Numpy Array
        target: (H, W) 0/1 Numpy Array
    """
    
    intersection = (pred * target).sum()
    total = pred.sum() + target.sum()
    
    # 如果兩張圖都是全黑 (沒有傷口)，Dice 應該是 1.0 (滿分)
    if total == 0:
        return 1.0
    
    return (2. * intersection) / (total + 1e-6)


def main():
    args = get_args()
    
    print(f"[INFO] Dataset:    {args.dataset}")
    print(f"[INFO] Split:      {args.split}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Device:     {args.device}")
    
    # 1. 載入模型
    if not os.path.exists(args.checkpoint):
        print(f"[Error] Checkpoint not found: {args.checkpoint}")
        return

    print("[INFO] Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(args.device)
    load_checkpoint(args.checkpoint, model)
    
    # 2. 準備 Dataset
    # 這裡我們只給 Resize，剩下的 Manual Normalization 交給 Dataset 內部處理
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    ])
    
    dataset = SegmentationDataset(
        root_dir=args.root,
        datasets=[args.dataset],
        split=args.split,
        transform=transform
    )
    
    if len(dataset) == 0:
        print(f"[Error] No images found for {args.dataset} ({args.split})")
        return
    
    # 3. 開始評估迴圈
    print(f"[INFO] Evaluating on {len(dataset)} images...")
    dice_scores = []
    
    # 這裡我們不使用 DataLoader，直接用 index 存取，確保一張一張算
    for i in tqdm(range(len(dataset))):
        img_tensor, mask_tensor = dataset[i]
        
        # A. 推論 (Prediction)
        pred_mask = infer_one_image(
            model,
            img_tensor,
            args.device,
            args.threshold
        )
        
        # B. 處理標準答案 (Ground Truth)
        # 把 Tensor 轉成 Numpy (H, W)，並確保它是整數 0/1
        gt_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
        
        # C. 算分
        score = calculate_dice(pred_mask, gt_mask)
        dice_scores.append(score)
    
    mean_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    print(f"\n📊 Evaluation Results")
    print(f"   Dataset:   {args.dataset}")
    print(f"   Mean Dice: {mean_dice:.4f}")
    print(f"   Std Dev:   {std_dice:.4f}")
    
    os.makedirs(args.out, exist_ok=True)
    
    report = {
        "dataset": args.dataset,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "mean_dice": float(mean_dice),
        "std_dice": float(std_dice),
        "num_samples": len(dataset),
        "scores_detail": [float(s) for s in dice_scores] # 存下每一張的分數
    }
    
    with open(args.out, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"✅ Report saved to {args.out}")


if __name__ == "__main__":
    main()