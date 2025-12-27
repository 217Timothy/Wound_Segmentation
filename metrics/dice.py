import torch

def dice_coeff(logits: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    計算 Dice Coefficient
    Args:
        logits: 模型輸出的原始數值 (B, 1, H, W)，尚未經過 Sigmoid
        target: 真實標籤 (B, 1, H, W)，值為 0 或 1
        smooth: 平滑項，避免除以 0
    Returns:
        dice: 計算出的 Dice 分數 (Scalar Tensor)
    """
    
    # 1. Apply Sigmoid (將 logits 轉為 0~1 機率)
    pred = torch.sigmoid(logits)
    
    # 2. Binarize (大於 0.5 設為 1，其餘為 0)
    # 使用 round 或 >0.5 都可以
    pred = (pred > 0.5).float()
    
    # 3. Flatten (拉平成一維向量以便計算)
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    # 4. 計算 Dice: (2 * 交集) / (總和)
    intersection = (pred_flat * target_flat).sum()
    
    # 注意：這裡不使用 .item()，因為我們要回傳 Tensor
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice