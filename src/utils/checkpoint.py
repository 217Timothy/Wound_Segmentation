import os
import torch
import torch.nn as nn
import shutil

def save_checkpoint(state, is_best, checkpoint_dir, filename='last.pt'):
    """
    儲存模型權重
    Args:
        state (dict): 要存的字典 (包含 model, optimizer, epoch, score)
        is_best (bool): 這一輪是否是目前表現最好的
        checkpoint_dir (str): 存檔資料夾路徑 (例如 'checkpoints/unet')
        filename (str): 檔名 (預設存成 last.pt)
    """
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best.pt")
        shutil.copy(filepath, best_path)
        print(f"[CheckPoint] ✅ New best model saved! Dice Score: {state.get('dice', 0):.4f}. IoU Score: {state.get('iou', 0): .4f}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """
    讀取模型權重
    """
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path}")
    
    print(f"[CheckPoint] Loading from {checkpoint_path} ...")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print(f"Last Run: Epoch: {checkpoint['epoch']}, Dice Score: {checkpoint['dice']: .4f}, IoU Score: {checkpoint['iou']: .4f}")
    
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    return checkpoint