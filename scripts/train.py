# ⭐ 只有「關鍵地方有註解 🔥」，其他都保留你原本

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
from collections import Counter  # 🔥 新增

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import UNet, ResUnet, EfficientUnet
from src.datasets import SegmentationDataset, TKRDataset, get_train_transforms, get_val_transforms  # 🔥 移除 TKR transform
from src.losses import BCEDiceLoss, BCETverskyLoss, FocalTverskyLoss
from src.utils.seed import seed_everything
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from src.engine import train_one_epoch, validate


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
PIN_MEMORY = torch.cuda.is_available()
DATA_ROOT_DIR = "data"


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


def load_pretrained_finetune(model, ckpt_path, device):
    if ckpt_path is None:
        return

    print(f"[INFO] Loading pretrained: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_dict = model.state_dict()

    matched = {}
    for k, v in state_dict.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    print(f"[INFO] Loaded {len(matched)} layers")


def build_dataset(dataset_names, split, transform, cache_data):
    return SegmentationDataset(
        root_dir=DATA_ROOT_DIR,
        datasets=dataset_names,
        split=split,
        transform=transform,
        cache_data=cache_data
    )


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
        model.model.segmentation_head = nn.Sequential(
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


def build_loss(name):
    if name == "bce_dice":
        return BCEDiceLoss()
    elif name == "bce_tversky":
        return BCETverskyLoss(0.4, 0.6)
    else:
        return FocalTverskyLoss(0.4, 0.6)


def main():
    args = get_args()

    seed_everything()

    print("[INFO] Mode:", "Finetune" if args.pretrained_ckpt else "Scratch")

    # 🔥 LR 防呆
    if args.pretrained_ckpt and args.lr > 5e-5:
        print("⚠️ LR too high for finetune")

    # ==========================================
    # Dataset
    # ==========================================
    train_ds = build_dataset(
        args.datasets,
        "train",
        get_train_transforms((IMAGE_SIZE, IMAGE_SIZE)),  # 🔥 修正
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

    # 🔥 dataset 分布
    if hasattr(train_ds, "files"):
        ds_names = [f[2] for f in train_ds.files]
        print("📊 Train Distribution:", Counter(ds_names))

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
    load_pretrained_finetune(model, args.pretrained_ckpt, DEVICE)

    if args.freeze_encoder_epochs > 0:
        freeze_encoder(model)

    loss_func = build_loss(args.loss_name)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_score = 0.0

    # ==========================================
    # Train Loop
    # ==========================================
    for epoch in range(1, args.epochs + 1):

        if epoch == args.freeze_encoder_epochs + 1:
            unfreeze_encoder(model)

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            loss_func,
            DEVICE,
            epoch,
            grad_clip=args.grad_clip
        )

        val_dict = validate(model, val_loader, loss_func, DEVICE)

        print(f"\nEpoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Dice: {val_dict['val_dice']:.4f}")

        # 🔥 ⭐ 核心升級
        if "per_dataset" in val_dict:
            print("\n📊 Per-Dataset Results:")
            for ds, m in val_dict["per_dataset"].items():
                print(f"{ds} → Dice: {m['dice']:.4f}")

        scheduler.step()

        if val_dict["val_dice"] > best_score:
            best_score = val_dict["val_dice"]

            save_checkpoint({
                "state_dict": model.state_dict(),
                "dice": best_score
            }, True, f"checkpoints/{args.run_name}")


if __name__ == "__main__":
    main()