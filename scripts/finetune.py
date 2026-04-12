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
        print(f"[INFO] Loading config: {known_args.config}")
        with open(known_args.config, "r") as f:
            defaults = yaml.safe_load(f)

    parser = argparse.ArgumentParser(parents=[conf_parser])

    # basic
    parser.add_argument("--version", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--datasets", type=str, nargs="+")

    # finetune
    parser.add_argument("--pretrained_ckpt", type=str)

    # training
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--freeze_epochs", type=int, default=5)

    # misc
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
    model = EfficientUnet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,   # finetune 時一定用 None
        decoder_attention_type="scse",
        classes=1
    ).to(DEVICE)
    return model


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
# Main
# ==========================================
def main():
    args = get_args()

    print(f"[INFO] Using device: {DEVICE}")
    print(f"[INFO] Version: {args.version}")
    print(f"[INFO] Run Name: {args.run_name}")
    print(f"[INFO] Datasets: {args.datasets}")
    print(f"[INFO] Pretrained: {args.pretrained_ckpt}")

    if args.pretrained_ckpt is None:
        raise ValueError("pretrained_ckpt must be provided for finetune.py")

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

    print(f"[INFO] Configuration saved to: {config_path}")

    # ==========================================
    # Logs / Checkpoints
    # ==========================================
    checkpoint_dir = os.path.join("checkpoints", args.version, args.run_name)
    log_dir = os.path.join("logs", args.version)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{args.run_name}.csv")
    if not os.path.exists(log_path):
        with open(log_path, mode="w", newline="") as f:
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
    # Dataset / DataLoader
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

    print(f"[INFO] Train size: {len(train_ds)}")
    print(f"[INFO] Val size: {len(val_ds)}")

    use_persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=use_persistent_workers,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=use_persistent_workers,
    )

    # ==========================================
    # Model / Loss / Optimizer / Scheduler / Scaler
    # ==========================================
    model = build_model()
    loss_func = build_loss(args.loss_name)

    # 載入 pretrained checkpoint
    checkpoint = load_checkpoint(args.pretrained_ckpt, model)

    # 載入後先驗證一次，確認這真的是你原本 best model
    print("\n[DEBUG] Validate BEFORE finetune")
    debug_val = validate(model, val_loader, loss_func, DEVICE)
    print(f"[DEBUG] Dice before finetune: {debug_val['val_dice']:.4f}")
    if "per_dataset" in debug_val:
        for ds, m in debug_val["per_dataset"].items():
            print(f"[DEBUG] {ds}: {m['dice']:.4f}")

    # 先 freeze encoder
    freeze_encoder(model)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(device="cuda", enabled=(DEVICE == "cuda"))

    best_dice = 0.0
    encoder_unfrozen = False

    # ==========================================
    # Train Loop
    # ==========================================
    for epoch in range(1, args.epochs + 1):

        # 到指定 epoch 後解凍 encoder
        if (
            (not encoder_unfrozen)
            and (epoch == args.freeze_epochs + 1)
            and (args.freeze_epochs > 0)
        ):
            unfreeze_encoder(model)
            encoder_unfrozen = True

            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=1e-4
            )

            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=args.epochs - epoch + 1,
                eta_min=1e-6
            )

            print("[INFO] Rebuilt optimizer/scheduler after unfreezing encoder.")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_func=loss_func,
            device=DEVICE,
            epoch=epoch,
            grad_clip=args.grad_clip
        )

        val_dict = validate(model, val_loader, loss_func, DEVICE)

        val_loss = val_dict["val_loss"]
        val_dice = val_dict["val_dice"]
        val_iou = val_dict["val_iou"]
        val_recall = val_dict["val_recall"]
        val_precision = val_dict["val_precision"]

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        new_lr = optimizer.param_groups[0]["lr"]

        if new_lr < current_lr:
            print(f"📉 [Scheduler] Learning Rate reduced from {current_lr:.2e} to {new_lr:.2e}")

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Val Recall: {val_recall:.4f} | "
            f"Val Precision: {val_precision:.4f}"
        )

        if "per_dataset" in val_dict:
            print("\n📊 Per-Dataset Results:")
            for ds, metrics in val_dict["per_dataset"].items():
                print(
                    f"{ds} → "
                    f"Dice: {metrics['dice']:.4f} | "
                    f"IoU: {metrics['iou']:.4f} | "
                    f"Recall: {metrics['recall']:.4f} | "
                    f"Precision: {metrics['precision']:.4f}"
                )

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

        # save log
        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                f"{epoch}",
                f"{train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{val_dice:.4f}",
                f"{val_iou:.4f}",
                f"{val_recall:.4f}",
                f"{val_precision:.4f}"
            ])

        # save checkpoint
        save_state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dice": val_dice,
            "iou": val_iou,
            "recall": val_recall,
            "precision": val_precision,
            "encoder_unfrozen": encoder_unfrozen
        }

        save_checkpoint(save_state, is_best, checkpoint_dir)

    print("\n✅ Finetune completed!")


if __name__ == "__main__":
    main()