"""End-to-end training driver shared by every training mode.

`run_training` is fully config-driven, so the entry-point scripts
(`scripts/train.py`, `scripts/finetune_wound.py`, `scripts/finetune_tkr.py`)
only have to choose a defaults preset and call this function.

Behavior controlled by the config:
    pretrained_ckpt        load matching weights before training (fine-tuning)
    freeze_encoder_epochs  keep the encoder frozen for the first N epochs
    resume                 continue from last.pt (skipped when fine-tuning)
    select_by              metric used to decide the "best" checkpoint
"""

from pathlib import Path

import torch.optim as optim
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader

from ..config import save_config
from ..data import WoundSegmentationDataset, build_train_transforms, get_val_transforms
from ..losses import build_loss
from ..models import build_model, model_config
from ..utils import (
    CSVLogger,
    get_device,
    load_checkpoint,
    load_weights_partial,
    save_checkpoint,
    seed_everything,
)
from .trainer import train_one_epoch
from .validator import validate

_LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "val_dice", "val_mean_dice",
                "val_iou", "val_recall", "val_precision"]


def set_encoder_trainable(model, trainable: bool) -> bool:
    """Freeze/unfreeze the encoder of an smp-based model.

    Returns True if an encoder was found (plain UNet has none).
    """
    encoder = getattr(getattr(model, "model", None), "encoder", None)
    if encoder is None:
        return False
    for p in encoder.parameters():
        p.requires_grad = trainable
    return True


def _build_loaders(cfg, device):
    img_size = (cfg.image_size, cfg.image_size)
    train_tf = build_train_transforms(cfg.augmentation, img_size)
    val_tf = get_val_transforms(img_size)

    train_ds = WoundSegmentationDataset(cfg.data_root, cfg.datasets, "train",
                                        train_tf, cfg.cache_data)
    val_ds = WoundSegmentationDataset(cfg.data_root, cfg.datasets, "val",
                                      val_tf, cfg.cache_data)
    if len(train_ds) == 0:
        raise RuntimeError(
            f"No training samples found in {cfg.data_root} for {cfg.datasets}.")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=pin)
    # batch size 1 keeps per-image validation simple and deterministic.
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=pin)
    return train_loader, val_loader


def run_training(cfg) -> Path:
    """Train (or fine-tune) a model end to end. Returns the best checkpoint path."""
    seed_everything(cfg.seed)
    device = get_device()
    print(f"[run] device={device}  version={cfg.version}  run={cfg.run_name}")

    train_loader, val_loader = _build_loaders(cfg, device)

    arch = model_config(cfg)
    model = build_model(encoder_weights=cfg.encoder_weights, device=device, **arch)
    if cfg.pretrained_ckpt:
        load_weights_partial(cfg.pretrained_ckpt, model, map_location=device)

    loss_func = build_loss(cfg.loss_name)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs,
                                                     eta_min=1e-6)
    scaler = GradScaler(enabled=cfg.amp and device == "cuda")

    out = Path(cfg.out_root)
    ckpt_dir = out / "checkpoints" / cfg.version / cfg.run_name
    run_dir = out / "runs" / cfg.version / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")
    logger = CSVLogger(out / "logs" / cfg.version / f"{cfg.run_name}.csv", _LOG_COLUMNS)

    # -- resume (only for fresh training, never when fine-tuning) ----------
    start_epoch, best_score = 1, 0.0
    last_ckpt = ckpt_dir / "last.pt"
    if cfg.resume and not cfg.pretrained_ckpt and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, model, optimizer, scheduler,
                               map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("metrics", {}).get(cfg.select_by, 0.0)
        print(f"[run] resumed at epoch {start_epoch} "
              f"(best {cfg.select_by}={best_score:.4f})")

    # -- optional encoder freezing -----------------------------------------
    encoder_frozen = False
    if cfg.freeze_encoder_epochs > 0:
        encoder_frozen = set_encoder_trainable(model, False)
        if encoder_frozen:
            print(f"[run] encoder frozen for the first "
                  f"{cfg.freeze_encoder_epochs} epoch(s)")

    # -- training loop ------------------------------------------------------
    for epoch in range(start_epoch, cfg.epochs + 1):
        if encoder_frozen and epoch > cfg.freeze_encoder_epochs:
            set_encoder_trainable(model, True)
            encoder_frozen = False
            print(f"[run] encoder unfrozen at epoch {epoch}")

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                     loss_func, device, epoch, cfg.grad_clip)
        val = validate(model, val_loader, loss_func, device)
        scheduler.step()

        print(f"\nepoch {epoch}/{cfg.epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  dice={val['dice']:.4f}  "
              f"mean_dice={val['mean_dice']:.4f}")
        for ds, m in val["per_dataset"].items():
            flag = "  <-- below 0.80" if m["dice"] < 0.80 else ""
            print(f"    {ds:<14} dice={m['dice']:.4f}  iou={m['iou']:.4f}  "
                  f"recall={m['recall']:.4f}  precision={m['precision']:.4f}{flag}")

        score = val[cfg.select_by]
        is_best = score > best_score
        if is_best:
            best_score = score
            print(f"[run] new best {cfg.select_by}={best_score:.4f}")

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": {"dice": val["dice"], "mean_dice": val["mean_dice"],
                        "iou": val["iou"], "loss": val["loss"]},
            "model_cfg": arch,
        }, is_best, ckpt_dir)

        logger.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val["loss"], 6),
            "val_dice": round(val["dice"], 6),
            "val_mean_dice": round(val["mean_dice"], 6),
            "val_iou": round(val["iou"], 6),
            "val_recall": round(val["recall"], 6),
            "val_precision": round(val["precision"], 6),
        })

    best_path = ckpt_dir / "best.pt"
    print(f"\n[run] finished. best {cfg.select_by}={best_score:.4f}")
    print(f"[run] best checkpoint: {best_path}")
    return best_path
