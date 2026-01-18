import os
import sys
import argparse
import csv
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import albumentations as A

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
BATCH_SIZE = 8       # 圖片大如果跑不動 (OOM)，改小成 2 或 1
NUM_WORKERS = 8      # Mac 建議設 0，Linux Server 可以設 4 或 8
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = True

# 資料路徑 (請確保你有這些資料夾)
DATA_ROOT_DIR = "data/processed"

def get_args():
    parser = argparse.ArgumentParser()
    
    # 必要參數：模型版本
    parser.add_argument("--version", type=str, required=True,
                        help="這次訓練的版本")
    parser.add_argument("--run_name", type=str, required=True,
                        help="這次訓練的名稱")
    
    # 其他設定
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=50,
                        help="輸入想訓練的epoch數")
    parser.add_argument("--dataset", type=str, nargs="+", default=["WoundSeg", "CO2Wound"],
                        help="輸入一個或多個資料集名稱 (用空白隔開)")
    
    return parser.parse_args()

def main():
    args = get_args()
    seed_everything()
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Using Device: {args.device}")
    print(f"[INFO] Using Datasets: {args.dataset}")
    
    
    checkpoint_dir = os.path.join("checkpoints", args.version, args.run_name)
    log_dir = os.path.join("logs", args.version)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.run_name}.csv"
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou"])
    
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
    print(f"[INFO] Loading Data from {DATA_ROOT_DIR} with datasets: {args.dataset}...")
    train_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=args.dataset,
        split="train",
        transform=train_transform
    )
    
    val_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=args.dataset,
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
        persistent_workers=True,
        shuffle=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=4,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        shuffle=False,
    )
    
    # 4. 初始化模型
    print("[INFO] Initializing Model...")
    model = UNet(n_channels=3, n_classes=1).to(args.device)
    compiled_model = model
    loss_func = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler(device="cuda", enabled=(args.device == "cuda"))
    if torch.cuda.is_available() and args.device == 'cuda':
        compiled_model = torch.compile(model, mode="reduce-overhead")
    
    checkpoint_path = os.path.join(checkpoint_dir, "last.pt")
    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model, optimizer)
    
    # 5. 開始訓練
    best_score = 0.0
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(compiled_model, train_loader=train_loader, optimizer=optimizer, scaler=scaler, loss_func=loss_func, device=args.device, epoch=epoch)
        
        val_dict = validate(compiled_model, val_loader, loss_func, args.device)
        val_loss = val_dict["val_loss"]
        val_dice = val_dict["val_dice"]
        val_iou = val_dict["val_iou"]
        
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f} | ")
        
        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice
        
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{epoch}", f"{train_loss: .4f}", f"{val_loss: .4f}", f"{val_dice: .4f}", f"{val_iou: .4f}"])
        
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dice": val_dice,
            "iou": val_iou
        }
        
        save_checkpoint(checkpoint, is_best, checkpoint_dir)


if __name__ == "__main__":
    main()