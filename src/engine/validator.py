import os
import sys
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.metrics.dice import calculate_dice
from src.metrics.iou import calculate_iou
from src.metrics.recall import calculate_recall


def validate(model, val_loader, loss_func, device):
    """
    執行驗證流程
    Returns:
        avg_loss: 平均 Loss
        avg_dice: 平均 Dice Score
    """
    
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_recall = 0.0
    
    with torch.no_grad():
        loop = tqdm(val_loader, desc="Validating")
        
        for imgs, masks in loop:
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # 1. 推論
            logits = model(imgs)
            
            # 2. 算 Loss (監控用)
            loss = loss_func(logits, masks)
            running_loss += loss.item()
            
            # 把 logits 轉乘 sigmoid，再轉乘預測值。用來計算 dice 與 iou
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # 3. 算 Dice
            dice = calculate_dice(preds, masks)
            running_dice += dice
            
            # 4. 算 IoU
            iou = calculate_iou(preds, masks)
            running_iou += iou
            
            # 5. 算 Recall
            recall = calculate_recall(preds, masks)
            running_recall += recall
            
            # 更新進度條
            loop.set_postfix(loss=f"{loss.item(): .4f}", dice=f"{dice: .4f}", iou=f"{iou: .4f}", recall=f"{recall: .4f}")
    
    avg_loss = running_loss / len(val_loader)
    avg_dice = running_dice / len(val_loader)
    avg_iou = running_iou / len(val_loader)
    avg_recall = running_recall / len(val_loader)
    
    return {
        "val_loss": avg_loss,
        "val_dice": avg_dice,
        "val_iou": avg_iou,
        "val_recall": avg_recall
    }