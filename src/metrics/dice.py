def calculate_dice(pred, target):
    """
    計算 Dice Coefficient
    
    Args:
        pred: 預測結果 (B, 1, H, W) 或 (B, H, W), 必須已經是 0/1 (Bool or Float)
        target: 真實標籤 (B, 1, H, W)，值為 0 或 1
    Returns:
        dice: 計算出的 Dice
    """
    
    # 1. Flatten
    preds_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 2. Intersection
    intersection = (preds_flat * target_flat).sum()
    
    # 3. Computing Dice Coeff: (2 * Intersection) / (預測總和 + 真實總和)
    dice_score = (2. * intersection) / (preds_flat.sum() + target_flat.sum() + 1e-6)
    
    # Return float (用 .item() 把 tensor 轉成數字)
    return dice_score.item()