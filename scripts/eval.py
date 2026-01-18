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
from src.engine import infer_one_image
from src.utils import load_checkpoint
from src.metrics.dice import calculate_dice
from src.metrics.iou import calculate_iou


# ==========================================
# 1. 設定參數與參數解析器
# ==========================================
IMAGE_SIZE = 512

def get_args():
    parser = argparse.ArgumentParser()
    
    # 必要參數：模型版本與資料集
    parser.add_argument("--version", type=str, required=True,
                        help="要使用的模型版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="第幾次跑")
    parser.add_argument("--datasets", type=str, required=True, nargs="+",
                        help="資料集名稱")
    
    # 輸入與輸出根目錄
    parser.add_argument("--in_root", type=str, default="data/processed",
                        help="輸入圖片的資料夾路徑")
    parser.add_argument("--out_root", type=str, default="results/run",
                        help="輸出結果的根目錄")
    
    # 其他設定
    parser.add_argument("--split", type=str, default="val",
                        help="要評估的清單 (test/val)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="使用設備")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二值化門檻")
    
    return parser.parse_args()


def main():
    args = get_args()
    
    checkpoint_path = os.path.join("checkpoints", args.version, args.run_name, "best.pt")
    output_json_dir = os.path.join(args.out_root, args.version)
    os.makedirs(output_json_dir, exist_ok=True)
    output_json_path = os.path.join(output_json_dir, f"metrics_{args.run_name}.json")
    
    print(f"[INFO] Dataset:    {args.datasets}")
    print(f"[INFO] Split:      {args.split}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device:     {args.device}")
    
    # 1. 載入模型
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return
    print("[INFO] Loading model...")
    model = UNet(n_channels=3, n_classes=1).to(args.device)
    load_checkpoint(checkpoint_path, model)
    
    # 2. 準備 Dataset
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    ])
    
    # 用來存最終結果的容器
    final_report = {}
    
    print(f"================ Evaluation Start ({args.split}) ================")
    
    # 3. 針對每一個 Dataset 跑迴圈
    for ds in args.datasets:
        try:
            dataset = SegmentationDataset(
                root_dir=args.in_root,
                datasets=[ds],
                split=args.split,
                transform=transform
            )
        except Exception as e:
            print(f"[Warn] Skip {ds}: {e}")
            continue
        
        if len(dataset) == 0:
            print(f"[Error] No images found for {ds} ({args.split})")
            return
        
        print(f"[INFO] Evaluating on {len(dataset)} images...")
        dice_scores = []
        iou_scores = []
        
        for i in tqdm(range(len(dataset)), desc=f"Evaluating {ds}"):
            img_tensor, mask_tensor = dataset[i]
            
            # A. 推論 (Prediction)
            pred_mask = infer_one_image(
                model,
                img_tensor,
                args.device,
                args.threshold
            )
            
            # B. 取得 GT (Ground Truth)
            # mask_tensor 是 (1, H, W)，我們轉成 numpy (H, W)
            gt_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
            
            # C. 轉成 Tensor 準備算分
            pred_tensor = torch.from_numpy(pred_mask).float()
            gt_tensor = torch.from_numpy(gt_mask).float()
            
            # D. 算 Dice 與 IoU
            dice = calculate_dice(pred_tensor, gt_tensor)
            iou = calculate_iou(pred_tensor, gt_tensor)
            
            dice_scores.append(dice)
            iou_scores.append(iou)
        
        # 統計該資料集的平均分
        mean_dice = np.mean(dice_scores)
        mean_iou = np.mean(iou_scores)
        
        print(f"   -> {ds}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}")
        
        final_report[ds] = {
            "mean_dice": float(mean_dice),
            "mean_iou": float(mean_iou),
            "samples": len(dataset)
        }
        
    # 4. 計算全部資料集的總平均 (All)
    if len(final_report) > 0:
        all_dice = np.mean([d["mean_dice"] for d in final_report.values()])
        all_iou = np.mean([d["mean_iou"] for d in final_report.values()])
        
        final_report["all"] = {
            "mean_dice": float(all_dice),
            "mean_iou": float(all_iou)
        }
        print(f"\n================ Summary ================")
        print(f" ALL DATASETS: Dice={all_dice:.4f}, IoU={all_iou:.4f}")
    
    with open(output_json_path, "w") as f:
        json.dump(final_report, f, indent=4)
    
    print(f"✅ Results saved to:\n  - {output_json_path}")


if __name__ == "__main__":
    main()