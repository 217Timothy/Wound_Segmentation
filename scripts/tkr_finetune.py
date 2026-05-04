import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler

from src.engine import train_one_epoch, validate
from src.utils.checkpoint import save_checkpoint

# ✅ 用你的 function
from train import (
    get_args,
    build_model,
    build_loss,
    build_dataset,
    freeze_encoder,
    unfreeze_encoder,
)

from src.datasets import get_tkr_finetune_train_transforms, get_val_transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 512
DATA_ROOT_DIR = "data_tkr"


# ==========================
# Load pretrained
# ==========================
def load_pretrained(model, path):
    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = ckpt["state_dict"]

    model_dict = model.state_dict()

    matched = {
        k: v for k, v in state_dict.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }

    model_dict.update(matched)
    model.load_state_dict(model_dict, strict=False)

    print(f"[TKR] Loaded {len(matched)} layers")


def main():
    args = get_args()

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Config: {args}")

    # ==========================
    # Dataset
    # ==========================
    train_ds = build_dataset(
        args.datasets,
        "train",
        get_tkr_finetune_train_transforms((IMAGE_SIZE, IMAGE_SIZE)),  # 🔥 TKR專用
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
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # ==========================
    # Model
    # ==========================
    model = build_model(args.version)

    # ==========================
    # Pretrained (finetune)
    # ==========================
    if args.pretrained_ckpt:
        load_pretrained(model, args.pretrained_ckpt)

    # ==========================
    # Loss
    # ==========================
    loss_func = build_loss(args.loss_name)

    # ==========================
    # Optimizer
    # ==========================
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )

    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_dice = 0

    # ==========================
    # Freeze encoder
    # ==========================
    if args.freeze_encoder_epochs > 0:
        freeze_encoder(model)

    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(1, args.epochs + 1):

        # 🔥 Unfreeze
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
            args.grad_clip
        )

        val_dict = validate(model, val_loader, loss_func, DEVICE)

        val_dice = val_dict["val_dice"]

        scheduler.step()

        print(
            f"[TKR] Epoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dice:.4f}"
        )

        is_best = val_dice > best_dice
        if is_best:
            best_dice = val_dice

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "dice": val_dice
        }, is_best, f"checkpoints/{args.version}/{args.run_name}")


if __name__ == "__main__":
    main()