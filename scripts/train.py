import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models.unet import UNet
from src.datasets import SegmentationDataset
from src.losses import BCEDiceLoss
from src.utils.seed import seed_everything
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engine import train_one_epoch, validate


# ==========================================
# 1. 設定參數 (Configuration)
# ==========================================
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4       # 圖片大如果跑不動 (OOM)，改小成 2 或 1
NUM_EPOCHS = 50      # 總共要跑幾輪
NUM_WORKERS = 0      # Mac 建議設 0，Linux Server 可以設 4 或 8
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False   # 如果要繼續訓練上次的進度，改成 True
RUN_NAME = "unet_v1"

# 資料路徑 (請確保你有這些資料夾)
DATA_ROOT_DIR = "data/processed"
DATASET_LIST = ["WoundSeg", "CO2Wound"]
CHECKPOINT_DIR = f"checkpoints/{RUN_NAME}/"


def main():
    seed_everything()
    print(f"[INFO] Using Device: {DEVICE}")
    
    # 2. 定義影像轉換 (Transforms)
    # 你的 Dataset 寫法裡，如果這裡傳入 transform，就會在 Dataset 內部被呼叫
    train_transform = A.Compose(transforms=[
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
    ])
    
    # 3. 準備資料集 (Dataset & DataLoader)
    print(f"[INFO] Loading Data from {DATA_ROOT_DIR} with datasets: {DATASET_LIST}...")
    train_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=DATASET_LIST,
        split="train",
        transform=train_transform
    )
    
    val_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=DATASET_LIST,
        split="val",
        transform=val_transform
    )
    
    print(f"✅ Training samples: {len(train_ds)}")
    print(f"✅ Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    
    # 4. 初始化模型
    print("[INFO] Initializing Model...")
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    loss_func = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    if LOAD_MODEL:
        load_checkpoint(os.path.join(CHECKPOINT_DIR, "last.pt"), model, optimizer)
    
    # 5. 開始訓練
    best_score = 0.0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_func, DEVICE, epoch)
        val_loss, val_dice = validate(model, val_loader, loss_func, DEVICE)
        
        print(f"Epoch [{epoch}/{NUM_EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | ")
        
        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice
        
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "score": val_dice
        }
        
        save_checkpoint(checkpoint, is_best, CHECKPOINT_DIR)


if __name__ == "__main__":
    main()