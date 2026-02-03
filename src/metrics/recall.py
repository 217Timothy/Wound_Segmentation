def calculate_recall(pred, target):
    """
    計算 Recall
    
    Args:
        pred: 預測結果 (B, 1, H, W) 或 (B, H, W), 必須已經是 0/1 (Bool or Float)
        target: 真實標籤 (B, 1, H, W)，值為 0 或 1
    Returns:
        recall: 計算出的 Recall
    """
    
    # 1. Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 2. 計算 TP, FN
    TP = (pred_flat * target_flat).sum()
    FN = ((1 - pred_flat) * target_flat).sum()
    
    # 3. Computing Recall: TP / (TP + FN)
    recall = TP / (TP + FN + 1e-6)
    
    return recall.item()