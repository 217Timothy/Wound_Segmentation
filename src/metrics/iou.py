def calculate_iou(pred, target):
    """
    計算 Intersection over Union (IoU)
    
    Args:
        pred: 預測結果 (B, 1, H, W) 或 (B, H, W), 必須是 0/1 (Bool or Float)
        target: 真實標籤 (B, 1, H, W) 或 (B, H, W), 必須是 0/1
    Returns:
        iou: 計算出的 IoU
    """

    # 1. Flatten
    preds_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 2. Calculate Intersection and Union
    intersection = (preds_flat * target_flat).sum()
    total = preds_flat.sum() + target_flat.sum()
    union = total - intersection
    
    # 3. Computing IoU:  
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    return iou.item()