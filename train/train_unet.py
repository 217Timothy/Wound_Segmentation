import sys
import os

# 確保可以引用到根目錄的模組 (如果你的執行路徑是在 train/ 底下這行很重要)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import csv

from datasets import WoundDataset
from models.unet import UNet
from utils import save_checkpoint, load_checkpoint
from metrics import dice_coeff


def main():
    # -----------------------------
    # 1. Basic settings
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")
    
    # -----------------------------
    # path
    # -----------------------------
    img_dir = "data/processed/images"
    mask_dir = "data/processed/masks"
    
    run_name = "unet_v1"
    ckpt_dir = f"outputs/checkpoints/{run_name}"
    log_dir = f"outputs/logs/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, "best.pt")
    log_path = os.path.join(log_dir, "train_log.csv")
    
    # -----------------------------
    # 2. Hyperparams
    # -----------------------------
    batch_size = 2
    num_epochs = 2
    lr = 1e-4
    num_workers = 4
    val_percent = 0.2
    
    # -----------------------------
    # 3. Dataset / DataLoader
    # -----------------------------
    full_dataset = WoundDataset(
        images_dir=img_dir,
        masks_dir=mask_dir,
        transform=None
    )
    
    n_val = int(len(full_dataset) * val_percent)
    n_train = len(full_dataset) - n_val
    
    train_set, val_set = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    
    print(f"Total: {len(full_dataset)} | Train: {len(train_set)} | Val: {len(val_set)}")
    
    # -----------------------------
    # 4. Model / Loss / Optimizer
    # -----------------------------
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # -----------------------------
    # Resume (optional)
    # -----------------------------
    start_epoch = 0
    best_dice: float = 0.0

    if os.path.exists(last_ckpt_path):
        print(f"[INFO] Resuming from {last_ckpt_path}")
        start_epoch, loaded_metric = load_checkpoint(last_ckpt_path, model, optimizer, device)
        # 注意：如果舊的 checkpoint 存的是 loss，這裡可能會讀錯，
        # 如果是新的訓練，建議先把 outputs/checkpoints 清空重跑
        if loaded_metric is not None:
            best_dice = loaded_metric 
        print(f"[INFO] start_epoch={start_epoch}, best_dice_so_far={best_dice:.4f}")
    
    # -----------------------------
    # 5. Training Loop
    # -----------------------------
    for epoch in range(start_epoch, num_epochs):
        
        # === Train ===
        model.train()
        epoch_loss = 0.0
        
        for step, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # (Optional) 可以在這裡印出 step loss
            # if step % 10 == 0: print(...)

        avg_train_loss = epoch_loss / len(train_loader)
        
        # === Validation (新增部分) ===
        model.eval()
        val_loss: float = 0.0
        val_dice_score: float = 0.0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                
                # Val Loss
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Val Dice (呼叫 metrics/dice.py)
                dice = dice_coeff(outputs, masks)
                val_dice_score += dice.item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice_score / len(val_loader)
        
        # Print Info
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Dice: {avg_val_dice:.4f}")
        
        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_dice])
        
        # Save Last
        save_checkpoint(last_ckpt_path, model, optimizer, epoch + 1, best_dice)
        
        # Save Best (根據 Dice 越大越好)
        if avg_val_dice > best_dice:
            print(f"🔥 Best Model Updated! ({best_dice:.4f} -> {avg_val_dice:.4f})")
            best_dice = avg_val_dice
            save_checkpoint(best_ckpt_path, model, optimizer, epoch + 1, best_dice)

    print("Done!")

if __name__ == "__main__":
    main()