import os
import sys
import argparse
import csv
import yaml
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
from src.losses import BCEDiceLoss, BCETverskyLoss
from src.utils.seed import seed_everything
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engine import train_one_epoch, validate


# ==========================================
# 1. 設定參數 (Configuration)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
PIN_MEMORY = True

# 資料路徑
DATA_ROOT_DIR = "data/processed"

def get_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None, help="Path to config file")
    known_args, remaining_args = conf_parser.parse_known_args()
    
    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        print(f"[INFO] Loading defaults from config: {known_args.config}")
        with open(known_args.config, mode='r') as f:
            defaults = yaml.safe_load(f)
    
    parser = argparse.ArgumentParser(parents=[conf_parser])
    
    # 必要參數：模型版本與資料集
    parser.add_argument("--version", type=str,
                        help="這次訓練的版本")
    parser.add_argument("--run_name", type=str,
                        help="這次訓練的名稱")
    parser.add_argument("--datasets", type=str, nargs="+", 
                        help="輸入一個或多個資料集名稱 (用空白隔開，如WoundSeg CO2Wound)")
    
    # Hyperparameter
    parser.add_argument("--epochs", type=int,
                        help="輸入想訓練的 epoch 數")
    parser.add_argument("--lr", type=float,
                        help="輸入想訓練的學習數")
    parser.add_argument("--batch_size", type=int,
                        help="輸入想訓練的 batch size")
    parser.add_argument("--num_workers", type=int,
                        help="輸入想訓練的 worker 數")
    
    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    
    return args

def main():
    args = get_args()
    seed_everything()
    torch.backends.cudnn.benchmark = True
    print(f"[INFO] Version: {args.version}")
    print(f"[INFO] Running Name: {args.run_name}")
    print(f"[INFO] Using Device: {DEVICE}")
    print(f"[INFO] Using Datasets: {args.datasets}")
    print(f"[INFO] Running Epochs: {args.epochs}")
    print(f"[INFO] Using Learning Rate: {args.lr}")
    print(f"[INFO] Using Batch Size: {args.batch_size}")
    print(f"[INFO] Using Number of Workers: {args.num_workers}")
    
    
    out_config_dir = os.path.join("results", "runs", args.version, args.run_name)
    os.makedirs(out_config_dir, exist_ok=True)
    config_path = os.path.join(out_config_dir, f"config.yaml")
    
    config_dict = vars(args)
    config_dict["model_class"] = "UNet"
    config_dict["loss_func"] = "BCETverskyLoss"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False, indent=4)

    print(f"[INFO] Configuration saved to: {config_path}\n")
    
    checkpoint_dir = os.path.join("checkpoints", args.version, args.run_name)
    log_dir = os.path.join("logs", args.version)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = f"{log_dir}/{args.run_name}.csv"
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_dice", "val_iou", "val_recall"])
    
    # 2. 定義影像轉換/預處理 (Transforms)
    # 你的 Dataset 寫法裡，如果這裡傳入 transform，就會在 Dataset 內部被呼叫
    train_transform = A.Compose(transforms=[
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        
        # 幾何變換
        A.Rotate(limit=35, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        
        # 色彩增強 (add in run3) 
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25), # (change to 0.15 and p to 0.25 in run 7. original: 0.2, p=0.5)
        A.RGBShift(r_shift_limit=15, g_shift_limit=5, b_shift_limit=5, p=0.25), # (change red to 15, blue and green to 5 and p to 0.25 in run 7. original: 20, p=0.5)
    ])
    
    val_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    ])
    
    # 3. 準備資料集 (Dataset & DataLoader)
    print(f"[INFO] Loading Data from {DATA_ROOT_DIR} with datasets: {args.datasets}...")
    train_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=args.datasets,
        split="train",
        transform=train_transform,
        cache_data=True
    )
    
    val_ds = SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=args.datasets,
        split="val",
        transform=val_transform,
        cache_data=True
    )
    
    print(f"✅ Training samples: {len(train_ds)}")
    print(f"✅ Validation samples: {len(val_ds)}\n")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=True,
        prefetch_factor=4,
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
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)
    compiled_model = model
    loss_func = BCETverskyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scaler = GradScaler(device="cuda", enabled=(DEVICE == "cuda"))
    if torch.cuda.is_available() and DEVICE == 'cuda':
        compiled_model = torch.compile(model, mode="reduce-overhead")
    
    
    best_score = 0.0
    start_epoch = 1
    
    best_checkpoint_path = os.path.join(checkpoint_dir, "best.pt")
    if os.path.exists(best_checkpoint_path):
        best_ckpt = torch.load(best_checkpoint_path, map_location="cpu")
        best_score = best_ckpt.get("dice", 0.0)
        print(f"[INFO] Detected existing best model with Dice: {best_score:.4f}")
    
    checkpoint_resume_path = os.path.join(checkpoint_dir, "last.pt")
    if os.path.exists(checkpoint_resume_path):
        print(f"[INFO] Resuming training from {checkpoint_resume_path}\n")
        load_checkpoint(checkpoint_resume_path, model, optimizer)
        last_ckpt = torch.load(checkpoint_resume_path, map_location="cpu")
        print(f"Last Run: Epoch: {last_ckpt['epoch']}, Dice Score: {last_ckpt['dice']: .4f}, IoU Score: {last_ckpt['iou']: .4f}\n")
        start_epoch = last_ckpt["epoch"] + 1
    
    # 5. 開始訓練
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(compiled_model, train_loader=train_loader, optimizer=optimizer, scaler=scaler, loss_func=loss_func, device=DEVICE, epoch=epoch)
        
        val_dict = validate(compiled_model, val_loader, loss_func, DEVICE)
        val_loss = val_dict["val_loss"]
        val_dice = val_dict["val_dice"]
        val_iou = val_dict["val_iou"]
        val_recall = val_dict["val_recall"]
        
        print(f"Epoch [{epoch}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Dice: {val_dice:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"Val Recall: {val_recall:.4f} | ")
        
        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice
        
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"{epoch}", f"{train_loss: .4f}", f"{val_loss: .4f}", f"{val_dice: .4f}", f"{val_iou: .4f}", f"{val_recall: .4f}"])
        
        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dice": val_dice,
            "iou": val_iou,
            "recall": val_recall
        }
        
        save_checkpoint(checkpoint, is_best, checkpoint_dir)


if __name__ == "__main__":
    main()