import os
import sys
import argparse
import csv
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import UNet, ResUnet, EfficientUnet
from src.datasets import SegmentationDataset, get_train_transforms, get_val_transforms
from src.losses import BCEDiceLoss, BCETverskyLoss, FocalTverskyLoss
from src.utils.seed import seed_everything
from src.utils.checkpoint import save_checkpoint
from src.engine import train_one_epoch, validate


# ==========================================
# Device
# ==========================================
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

IMAGE_SIZE = 512
PIN_MEMORY = torch.cuda.is_available()
DATA_ROOT_DIR = "data"


# ==========================================
# Args
# ==========================================
def get_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None)
    known_args, remaining_args = conf_parser.parse_known_args()

    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument("--version", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--datasets", type=str, nargs="+")

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--cache_data", action="store_true")
    parser.add_argument("--loss_name", type=str, default="focal_tversky")

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    return args


# ==========================================
# Freeze / Unfreeze
# ==========================================
def freeze_encoder(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        for p in model.model.encoder.parameters():
            p.requires_grad = False
        print("[INFO] Encoder frozen")


def unfreeze_encoder(model):
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        for p in model.model.encoder.parameters():
            p.requires_grad = True
        print("[INFO] Encoder unfrozen")


# ==========================================
# Dataset
# ==========================================
def build_dataset(dataset_names, split, transform, cache_data):
    return SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=dataset_names,
        split=split,
        transform=transform,
        cache_data=cache_data
    )


# ==========================================
# Model
# ==========================================
def build_model(version):
    if version == "v1":
        return UNet(3, 1).to(DEVICE)

    elif version == "v2":
        model = ResUnet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

        old_head = model.model.segmentation_head
        model.model.segmentation_head = nn.Sequential(  # type: ignore
            nn.Dropout2d(0.3),
            old_head
        )
        return model

    elif version == "v3":
        return EfficientUnet(
            encoder_name="efficientnet-b3",
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            classes=1
        ).to(DEVICE)

    else:
        raise ValueError(f"Unsupported version: {version}")


# ==========================================
# Loss
# ==========================================
def build_loss(name):
    if name == "bce_dice":
        return BCEDiceLoss()
    elif name == "bce_tversky":
        return BCETverskyLoss(0.4, 0.6)
    elif name == "focal_tversky":
        return FocalTverskyLoss(0.4, 0.6)
    else:
        raise ValueError(f"Unsupported loss_name: {name}")


# ==========================================
# Optimizer
# ==========================================
# def build_optimizer(model, lr):
#     return optim.AdamW([
#         {"params": model.model.encoder.parameters(), "lr": lr * 0.1},
#         {"params": model.model.decoder.parameters(), "lr": lr},
#         {"params": model.model.segmentation_head.parameters(), "lr": lr},
#     ], weight_decay=1e-4)


# ==========================================
# Main
# ==========================================
def main():
    args = get_args()

    seed_everything()
    print(f"[INFO] Using device: {DEVICE}")

    # ==========================================
    # Save config
    # ==========================================
    out_config_dir = os.path.join("results", "runs", args.version, args.run_name)
    os.makedirs(out_config_dir, exist_ok=True)

    config_path = os.path.join(out_config_dir, "config.yaml")
    config_dict = vars(args)
    config_dict["device"] = DEVICE

    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, sort_keys=False, indent=4)

    # ==========================================
    # Logs / Checkpoints
    # ==========================================
    checkpoint_dir = os.path.join("checkpoints", args.version, args.run_name)
    log_dir = os.path.join("logs", args.version)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{args.run_name}.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "val_loss",
                "val_dice",
                "val_iou",
                "val_recall",
                "val_precision"
            ])

    # ==========================================
    # Dataset
    # ==========================================
    train_ds = build_dataset(
        args.datasets, "train",
        get_train_transforms((IMAGE_SIZE, IMAGE_SIZE)),
        args.cache_data
    )

    val_ds = build_dataset(
        args.datasets, "val",
        get_val_transforms((IMAGE_SIZE, IMAGE_SIZE)),
        args.cache_data
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )

    # ==========================================
    # Model
    # ==========================================
    model = build_model(args.version)
    loss_func = build_loss(args.loss_name)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    # ==========================================
    # Resume
    # ==========================================
    last_ckpt_path = os.path.join(checkpoint_dir, "last.pt")

    start_epoch = 1
    best_score = 0.0

    if os.path.exists(last_ckpt_path):
        print(f"[INFO] Resuming from: {last_ckpt_path}")

        ckpt = torch.load(last_ckpt_path, map_location=DEVICE)

        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])

        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("dice", 0.0)

        print(f"[INFO] Resume from epoch {start_epoch}")

    # ==========================================
    # Train Loop
    # ==========================================
    for epoch in range(start_epoch, args.epochs + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_func,
            DEVICE,
            epoch,
            args.grad_clip
        )

        val_dict = validate(model, val_loader, loss_func, DEVICE)

        val_loss = val_dict["val_loss"]
        val_dice = val_dict["val_dice"]
        val_iou = val_dict["val_iou"]
        val_recall = val_dict["val_recall"]
        val_precision = val_dict["val_precision"]

        scheduler.step()

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice

        checkpoint = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dice": val_dice,
        }

        save_checkpoint(checkpoint, is_best, checkpoint_dir)


if __name__ == "__main__":
    main()