import os
import sys
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.metrics.dice import dice_coeff


def validate(model, data_loader, loss_func, device):
    """
    執行驗證流程
    Returns:
        avg_loss: 平均 Loss
        avg_dice: 平均 Dice Score
    """
    
    model.eval()
    
    running_loss = 0.0
    running_dice = 0.0
    
    with torch.no_grad():
        loop = tqdm(data_loader, desc="Validating")
        
        for batch_idx, (imgs, masks) in enumerate(loop):
            imgs = imgs.to(device)
            masks = masks.to(device)
            
            # 1. 推論
            preds = model(imgs)
            
            # 2. 算 Loss (監控用)
            loss = loss_func(preds, masks)
            running_loss += loss.item()
            
            # 3. 算 Dice
            dice = dice_coeff(preds, masks)
            running_dice += dice
            
            # 更新進度條
            loop.set_postfix(loss=loss.item(), dice=dice)
    
    avg_loss = running_loss / len(data_loader)
    avg_dice = running_dice / len(data_loader)
    
    return avg_loss, avg_dice