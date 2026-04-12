import os
import sys
import argparse
import csv
import yaml
import torch
import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import EfficientUnet
from src.datasets import SegmentationDataset, get_train_transforms, get_val_transforms
from src.losses import FocalTverskyLoss, BCETverskyLoss, BCEDiceLoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engine import train_one_epoch, validate


# ==========================================
# Device
# ==========================================
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

IMAGE_SIZE = 512
PIN_MEMORY = torch.cuda.is_available()
DATA_ROOT_DIR = "data"


# ==========================================
# Args (支援 config.yaml)
# ==========================================
def get_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config", type=str, default=None)

    known_args, remaining_args = conf_parser.parse_known_args()

    defaults = {}
    if known_args.config and os.path.exists(known_args.config):
        print(f"[INFO] Loading config: {known_args.config}")
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    parser.add_argument("--version", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--pretrained_ckpt", type=str)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--lr", type=float)

    parser.add_argument("--freeze_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--cache_data", action="store_true")
    parser.add_argument("--loss_name", type=str, default="focal_tversky")

    parser.set_defaults(**defaults)
    args = parser.parse_args(remaining_args)
    return args


# ==========================================
# Model
# ==========================================
def build_model():
    return EfficientUnet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,
        decoder_attention_type="scse",
        classes=1
    ).to(DEVICE)


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
        raise ValueError(f"Unsupported loss: {name}")


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
# Freeze / Unfreeze
# ==========================================
def freeze_encoder(model):
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    print("[INFO] Encoder frozen")


def unfreeze_encoder(model):
    for p in model.model.encoder.parameters():
        p.requires_grad = True
    print("[INFO] Encoder unfrozen")


# ==========================================
# Main
# ==========================================
def main():
    args = get_args()

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Datasets: {args.datasets}")

    # checkpoint / log
    checkpoint_dir = os.path.join("checkpoints", args.version, args.run_name)
    log_dir = os.path.join("logs", args.version)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{args.run_name}.csv")

    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "val_loss",
                "global_dice", "mean_dice"
            ])

    # ==========================================
    # Dataset
    # ==========================================
    train_ds = build_dataset(
        args.datasets,
        "train",
        get_train_transforms((IMAGE_SIZE, IMAGE_SIZE)),
        args.cache_data
    )

    val_ds = build_dataset(
        args.datasets,
        "val",
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
    model = build_model()
    loss_func = build_loss(args.loss_name)

    load_checkpoint(args.pretrained_ckpt, model)

    # DEBUG（確認沒壞）
    print("\n[DEBUG] Before Finetune")
    debug_val = validate(model, val_loader, loss_func, DEVICE)
    print(f"[DEBUG] Global Dice: {debug_val['val_dice']:.4f}")

    # ==========================================
    # Optimizer
    # ==========================================
    freeze_encoder(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_score = 0.0
    encoder_unfrozen = False

    # ==========================================
    # Train Loop
    # ==========================================
    for epoch in range(1, args.epochs + 1):

        # unfreeze
        if epoch == args.freeze_epochs + 1 and not encoder_unfrozen:
            unfreeze_encoder(model)
            encoder_unfrozen = True

            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # train
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

        # validate
        val_dict = validate(model, val_loader, loss_func, DEVICE)

        global_dice = val_dict["val_dice"]

        # 🔥 核心：mean per-dataset
        per_ds = val_dict["per_dataset"]
        mean_dice = sum([v["dice"] for v in per_ds.values()]) / len(per_ds)

        print(f"\nEpoch {epoch}")
        print(f"Global Dice: {global_dice:.4f}")
        print(f"Mean Per-Dataset Dice: {mean_dice:.4f}")
        print("\n📊 Per-Dataset Results:")
        for ds, metrics in per_ds.items():
            print(
                f"{ds} → "
                f"Dice: {metrics['dice']:.4f} | "
                f"IoU: {metrics['iou']:.4f} | "
                f"Recall: {metrics['recall']:.4f} | "
                f"Precision: {metrics['precision']:.4f}"
    )

        # ==========================================
        # 存 checkpoint（用 mean）
        # ==========================================
        is_best = mean_dice > best_score
        if is_best:
            best_score = mean_dice

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "dice": mean_dice
        }, is_best, checkpoint_dir)

        # log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_loss,
                val_dict["val_loss"],
                global_dice,
                mean_dice
            ])

        scheduler.step()

    print("\n✅ Finetune Done!")


if __name__ == "__main__":
    main()