import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from src.models import EfficientUnet
from src.datasets import SegmentationDataset, get_train_transforms, get_val_transforms
from src.engine import train_one_epoch, validate
from src.losses import FocalTverskyLoss


# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
IMAGE_SIZE = 512


# =========================
# Args (CONFIG SUPPORT)
# =========================
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

    parser.add_argument("--datasets", type=str, nargs="+")
    parser.add_argument("--pretrained_ckpt", type=str)

    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)

    parser.add_argument("--lr", type=float)
    parser.add_argument("--freeze_epochs", type=int)

    parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_args)
    return args


# =========================
# Model
# =========================
def build_model():
    model = EfficientUnet(
        encoder_name="efficientnet-b3",
        encoder_weights=None,   # 🔥 MUST be None for finetune
        decoder_attention_type="scse",
        classes=1
    ).to(DEVICE)

    return model


# =========================
# Load checkpoint (STRICT)
# =========================
def load_checkpoint_strict(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    print(f"[INFO] Loading checkpoint: {path}")

    ckpt = torch.load(path, map_location=DEVICE)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    model.load_state_dict(state_dict, strict=True)
    print("[INFO] ✅ Loaded checkpoint (strict=True)")


# =========================
# Freeze / Unfreeze
# =========================
def freeze_encoder(model):
    for p in model.model.encoder.parameters():
        p.requires_grad = False
    print("[INFO] Encoder frozen")


def unfreeze_encoder(model):
    for p in model.model.encoder.parameters():
        p.requires_grad = True
    print("[INFO] Encoder unfrozen")


# =========================
# Main
# =========================
def main():
    args = get_args()

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Datasets: {args.datasets}")

    # =========================
    # Dataset
    # =========================
    train_ds = SegmentationDataset(
        root_dir="data",
        datasets=args.datasets,
        split="train",
        transform=get_train_transforms((IMAGE_SIZE, IMAGE_SIZE)),
    )

    val_ds = SegmentationDataset(
        root_dir="data",
        datasets=args.datasets,
        split="val",
        transform=get_val_transforms((IMAGE_SIZE, IMAGE_SIZE)),
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

    print(f"[INFO] Train size: {len(train_ds)}")
    print(f"[INFO] Val size: {len(val_ds)}")

    # =========================
    # Model
    # =========================
    model = build_model()

    # =========================
    # Load pretrained
    # =========================
    load_checkpoint_strict(model, args.pretrained_ckpt)

    # 🔥 VERY IMPORTANT DEBUG STEP
    print("\n[DEBUG] Validate BEFORE finetune")
    debug_val = validate(model, val_loader, FocalTverskyLoss(0.4, 0.6), DEVICE)

    print(f"[DEBUG] Dice before finetune: {debug_val['val_dice']:.4f}")
    if "per_dataset" in debug_val:
        for ds, m in debug_val["per_dataset"].items():
            print(f"[DEBUG] {ds}: {m['dice']:.4f}")

    # =========================
    # Freeze encoder
    # =========================
    freeze_encoder(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_func = FocalTverskyLoss(0.4, 0.6)

    # =========================
    # Training Loop
    # =========================
    for epoch in range(1, args.epochs + 1):

        # 🔥 Unfreeze
        if epoch == args.freeze_epochs + 1:
            unfreeze_encoder(model)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
            print("[INFO] Rebuild optimizer after unfreeze")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            None,
            loss_func,
            DEVICE,
            epoch
        )

        val_dict = validate(model, val_loader, loss_func, DEVICE)

        print(
            f"\nEpoch {epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Dice: {val_dict['val_dice']:.4f}"
        )

        if "per_dataset" in val_dict:
            for ds, m in val_dict["per_dataset"].items():
                print(f"{ds}: Dice={m['dice']:.4f}")

    print("\n✅ Finetune completed!")


if __name__ == "__main__":
    main()