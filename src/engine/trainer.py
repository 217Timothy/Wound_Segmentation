import torch
from torch.amp.autocast_mode import autocast
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, scaler, loss_func, device, epoch, grad_clip=None):
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
    
    device_type = "cuda" if str(device).startswith("cuda") else "cpu"
    use_amp = device_type == "cuda"
    
    loop = tqdm(train_loader, desc=f"Training Epoch: {epoch}")
    
    for imgs, masks, _ in loop:
        # 1. 把資料搬到 GPU
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        # 2. 清空梯度 (重要！)
        optimizer.zero_grad(set_to_none=True)
        
        # 3. 前向傳播 (Forward): 模型預測
        with autocast(device_type=device, enabled=use_amp):
            logits = model(imgs)
            # 4. 計算誤差 (Loss)
            loss = loss_func(logits, masks)
        
        # 5. 反向傳播 (Backward): 算出要怎麼修正
        scaler.scale(loss).backward() # 原：loss.backward()
        
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # 6. 更新參數 (Step): 實際修正模型
        scaler.step(optimizer) # 原：optimizer.step()
        scaler.update()
        
        # --- 紀錄數據 ---
        running_loss += loss.item()
        # 更新進度條後面的資訊 (即時看到 Loss 變化)
        loop.set_postfix(loss=loss.item())
    
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss