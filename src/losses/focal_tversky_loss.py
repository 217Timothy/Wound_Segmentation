import torch
import torch.nn as nn



class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        """
        Args:
            alpha: 控制 FP 的權重
            beta:  控制 FN 的權重
            smooth: 平滑項
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, logits, target):
        """
        Args:
            logits (_type_): _description_
            target (_type_): _description_
        """
        
        # 1. 轉成機率
        target = target.float()
        probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        
        # 2. 攤平 (Flatten) and 算 TP, FP, FN
        # view(-1) 會把所有維度拉成一條長長的向量
        dims = (1, 2, 3)  # (C,H,W) -> 每個 batch 各自算
        TP = (probs * target).sum(dim=dims)
        FP = (probs * (1 - target)).sum(dim=dims)
        FN = ((1 - probs) * target).sum(dim=dims)
        
        tversky_score = (TP + self.smooth) / (TP + (self.alpha * FP) + (self.beta * FN) + self.smooth)
        tversky_loss = 1 - tversky_score
        return tversky_loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.gamma = gamma
        
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    
    def forward(self, logits, target):
        loss_tversky = self.tversky(logits, target)
        focal_tversky_loss = torch.pow(loss_tversky, self.gamma)
        return focal_tversky_loss.mean()