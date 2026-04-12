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
# Load pretrained for finetune
# ==========================================
def load_pretrained_finetune(model, ckpt_path, device):
    if ckpt_path is None:
        print("[INFO] No pretrained checkpoint provided.")
        return

    if not os.path.exists(ckpt_path):
        print(f"[WARN] Pretrained checkpoint not found: {ckpt_path}")
        return

    print(f"[INFO] Loading pretrained: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_dict = model.state_dict()

    matched = {}
    skipped = []

    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
        else:
            skipped.append(k)

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    print(f"[INFO] Loaded {len(matched)} matched layers.")
    if len(skipped) > 0:
        print(f"[INFO] Skipped {len(skipped)} unmatched layers.")


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
    print(f"[INFO] Random seed set to 42")
    print(f"[INFO] Using device: {DEVICE}")

    if args.pretrained_ckpt is not None:
        print("[INFO] Mode: Finetune")
    else:
        print("[INFO] Mode: Scratch")

    if args.pretrained_ckpt and args.lr > 5e-5:
        print("⚠️ LR too high for finetune")

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

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")

    if hasattr(train_ds, "files"):
        ds_names = [f[2] for f in train_ds.files]
        print("📊 Train Distribution:", Counter(ds_names))

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
    # Model / Optimizer / Scheduler / Scaler
    # ==========================================
    model = build_model(args.version)
    loss_func = build_loss(args.loss_name)

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

    # ==========================================
    # Resume or Finetune
    # ==========================================
    last_ckpt_path = os.path.join(checkpoint_dir, "last.pt")
    best_ckpt_path = os.path.join(checkpoint_dir, "best.pt")

    start_epoch = 1
    best_score = 0.0
    encoder_unfrozen = False

    # 先讀 best score
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu")
        best_score = best_ckpt.get("dice", 0.0)
        print(f"[INFO] Existing best Dice: {best_score:.4f}")

    # 優先 resume
    if os.path.exists(last_ckpt_path):
        print(f"[INFO] Resuming from: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=DEVICE)

        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])

        start_epoch = ckpt["epoch"] + 1
        best_score = max(best_score, ckpt.get("dice", 0.0))

        # 若 checkpoint 有記錄 scheduler，就一併恢復
        if "scheduler" in ckpt:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e:
                print(f"[WARN] Failed to load scheduler state: {e}")

        # 若 checkpoint 有記錄 encoder 是否已解凍，也恢復
        if "encoder_unfrozen" in ckpt:
            encoder_unfrozen = ckpt["encoder_unfrozen"]

        print(f"[INFO] Resume start epoch: {start_epoch}")

    else:
        # 沒有 last.pt 才走 finetune
        load_pretrained_finetune(model, args.pretrained_ckpt, DEVICE)

        if args.freeze_encoder_epochs > 0:
            freeze_encoder(model)
            encoder_unfrozen = False

    compiled_model = model
    if torch.cuda.is_available() and DEVICE == "cuda":
        compiled_model = torch.compile(model, mode="reduce-overhead")

    # ==========================================
    # Train Loop
    # ==========================================
    for epoch in range(start_epoch, args.epochs + 1):

        # 到指定 epoch 後解凍 encoder
        if (
            (not encoder_unfrozen)
            and (epoch > args.freeze_encoder_epochs)
            and (args.freeze_encoder_epochs > 0)
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
            compiled_model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            loss_func=loss_func,
            device=DEVICE,
            epoch=epoch,
            grad_clip=args.grad_clip
        )

        val_dict = validate(compiled_model, val_loader, loss_func, DEVICE)

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

        is_best = val_dice > best_score
        if is_best:
            best_score = val_dice

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

        checkpoint = {
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

        save_checkpoint(checkpoint, is_best, checkpoint_dir)


if __name__ == "__main__":
    main()