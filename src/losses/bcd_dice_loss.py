import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, target):
        """
        Args:
            logits: (B, 1, H, W) 模型輸出 (還沒過 Sigmoid)
            targets: (B, 1, H, W) 真實標籤 (0 或 1)
        """
        
        # 1. 轉成機率
        probs = torch.sigmoid(logits)
        
        # 2. 攤平 (Flatten)
        # view(-1) 會把所有維度拉成一條長長的向量
        probs_flat = probs.view(-1)
        target_flat = target.view(-1)
        
        # 3. 計算交集 (Intersection)
        intersection = (probs_flat * target_flat).sum()
        
        # 4. Dice 公式: (2 * 交集) / (預測總和 + 真實總和)
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + target_flat.sum() + self.smooth)
        
        # 5. 回傳 Loss (1 - Score)
        return (1 - dice_score)


class BCEDiceLoss(nn.Module):
    """
    結合 BCE (像素級分類精準度) + Dice (整體形狀重疊率)
    """
    
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, logits, target):
        loss_bce = self.bce(logits, target)
        loss_dice = self.dice(logits, target)
        
        return self.bce_weight * loss_bce + (1 - self.bce_weight) * loss_dice