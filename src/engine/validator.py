import os
import sys
import torch
from tqdm import tqdm
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.metrics import calculate_dice, calculate_iou, calculate_recall, calculate_precision


def validate(model, val_loader, loss_func, device):
    
    model.eval()
    
    # ===== overall =====
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_recall = 0.0
    running_precision = 0.0
    
    # ===== per dataset =====
    ds_stats = defaultdict(lambda: {
        "count": 0,
        "dice": 0.0,
        "iou": 0.0,
        "recall": 0.0,
        "precision": 0.0
    })
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validating")
        
        for imgs, masks, ds_names in loop:   # 🔥 改這裡
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            logits = model(imgs)
            
            loss = loss_func(logits, masks)
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            dice = calculate_dice(preds, masks)
            iou = calculate_iou(preds, masks)
            recall = calculate_recall(preds, masks)
            precision = calculate_precision(preds, masks)
            
            # ===== overall =====
            running_dice += dice
            running_iou += iou
            running_recall += recall
            running_precision += precision
            
            # ===== per dataset =====
            for i in range(len(ds_names)):
                ds = ds_names[i]
                
                ds_stats[ds]["count"] += 1
                ds_stats[ds]["dice"] += dice
                ds_stats[ds]["iou"] += iou
                ds_stats[ds]["recall"] += recall
                ds_stats[ds]["precision"] += precision
            
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice:.4f}"
            )
    
    # ===== overall =====
    avg_loss = running_loss / len(val_loader)
    avg_dice = running_dice / len(val_loader)
    avg_iou = running_iou / len(val_loader)
    avg_recall = running_recall / len(val_loader)
    avg_precision = running_precision / len(val_loader)
    
    # ===== per dataset 平均 =====
    ds_results = {}
    
    print("\n📊 Per-Dataset Metrics:")
    for ds, stats in ds_stats.items():
        count = stats["count"]
        
        if count == 0:
            continue
        
        ds_results[ds] = {
            "dice": stats["dice"] / count,
            "iou": stats["iou"] / count,
            "recall": stats["recall"] / count,
            "precision": stats["precision"] / count
        }
        
        print(f"  {ds}: Dice={ds_results[ds]['dice']:.4f} | IoU={ds_results[ds]['iou']:.4f}")
    
    return {
        "val_loss": avg_loss,
        "val_dice": avg_dice,
        "val_iou": avg_iou,
        "val_recall": avg_recall,
        "val_precision": avg_precision,
        "per_dataset": ds_results   # 🔥 新增
    }