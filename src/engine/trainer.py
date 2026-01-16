import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, loss_func, device, epoch):
    """
    執行一個 Epoch 的訓練
    Args:
        model: 模型
        loader: 訓練資料載入器 (Train DataLoader)
        optimizer: 優化器 (Adam)
        loss_fn: 損失函數 (BCEDiceLoss)
        device: CPU 或 GPU
        epoch: 當前是第幾輪 (用來顯示在進度條上)
    Returns:
        epoch_loss: 這一輪的平均 Loss
    """
    
    model.train()
    
    running_loss = 0.0
    
    loop = tqdm(train_loader, desc=f"Training Epoch: {epoch}")
    
    for batch_idx, (imgs, masks) in enumerate(loop):
        # 1. 把資料搬到 GPU
        imgs = imgs.to(device)
        masks = masks.to(device)
        
        # 2. 清空梯度 (重要！)
        optimizer.zero_grad()
        
        # 3. 前向傳播 (Forward): 模型預測
        preds = model(imgs)
        
        # 4. 計算誤差 (Loss)
        loss = loss_func(preds, masks)
        
        # 5. 反向傳播 (Backward): 算出要怎麼修正
        loss.backward()
        
        # 6. 更新參數 (Step): 實際修正模型
        optimizer.step()
        
        # --- 紀錄數據 ---
        running_loss += loss.item()
        
        # 更新進度條後面的資訊 (即時看到 Loss 變化)
        loop.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss