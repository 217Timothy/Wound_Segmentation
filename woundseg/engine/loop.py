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

import json
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

    skip_empty_masks = cfg.get("skip_empty_masks", False)
    train_ds = WoundSegmentationDataset(
        cfg.data_root, cfg.datasets, "train", train_tf, cfg.cache_data,
        skip_empty_masks=skip_empty_masks)
    val_ds = WoundSegmentationDataset(
        cfg.data_root, cfg.datasets, "val", val_tf, cfg.cache_data,
        skip_empty_masks=skip_empty_masks)
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


def _clean_metrics(metrics: dict) -> dict:
    """Convert metric values to plain JSON-friendly Python numbers."""
    out = {}
    for key, value in metrics.items():
        if key == "per_dataset":
            out[key] = {
                ds: {m: float(v) for m, v in ds_metrics.items()}
                for ds, ds_metrics in value.items()
            }
        elif key == "loss":
            out["val_loss"] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _epoch_record(epoch: int, train_loss: float, val: dict, score: float,
                  is_best: bool) -> dict:
    return {
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "score": float(score),
        "is_best": bool(is_best),
        **_clean_metrics(val),
    }


def _public_epoch_record(record: dict | None) -> dict | None:
    if record is None:
        return None
    return {k: v for k, v in record.items() if k != "is_best"}


def run_training(cfg, epoch_callback=None) -> dict:
    """Train (or fine-tune) a model end to end. Returns a run summary dict.

    epoch_callback (optional): 物件需有 .on_epoch_start() 和 .on_epoch_end(epoch, metrics)。
        傳 None 時行為跟原本完全一樣，給計時/監控用。
    """
    seed_everything(cfg.seed)
    device = get_device()
    print(f"[run] device={device}  version={cfg.version}  run={cfg.run_name}")

    train_loader, val_loader = _build_loaders(cfg, device)

    arch = model_config(cfg)
    model = build_model(encoder_weights=cfg.encoder_weights, device=device, **arch)
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

    # -- resume / initialize weights ---------------------------------------
    start_epoch, best_score = 1, 0.0
    last_ckpt = ckpt_dir / "last.pt"
    if cfg.resume and last_ckpt.exists():
        ckpt = load_checkpoint(last_ckpt, model, optimizer, scheduler,
                               map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_score = ckpt.get("metrics", {}).get(cfg.select_by, 0.0)
        print(f"[run] resumed at epoch {start_epoch} "
              f"(best {cfg.select_by}={best_score:.4f})")
    elif cfg.pretrained_ckpt:
        load_weights_partial(cfg.pretrained_ckpt, model, map_location=device)

    # -- optional encoder freezing -----------------------------------------
    encoder_frozen = False
    if cfg.freeze_encoder_epochs > 0:
        encoder_frozen = set_encoder_trainable(model, False)
        if encoder_frozen:
            print(f"[run] encoder frozen for the first "
                  f"{cfg.freeze_encoder_epochs} epoch(s)")

    # -- training loop ------------------------------------------------------
    history = []
    best_record = None
    final_record = None
    for epoch in range(start_epoch, cfg.epochs + 1):
        if epoch_callback is not None:
            epoch_callback.on_epoch_start()

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

        record = _epoch_record(epoch, train_loss, val, score, is_best)
        history.append(record)
        final_record = record
        if is_best:
            best_record = record

        save_checkpoint({
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": {"dice": val["dice"], "mean_dice": val["mean_dice"],
                        "iou": val["iou"], "recall": val["recall"],
                        "precision": val["precision"], "loss": val["loss"],
                        "per_dataset": val["per_dataset"]},
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

        if epoch_callback is not None:
            epoch_callback.on_epoch_end(epoch, {
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_dice": val["dice"],
                "val_mean_dice": val["mean_dice"],
            })

    best_path = ckpt_dir / "best.pt"
    print(f"\n[run] finished. best {cfg.select_by}={best_score:.4f}")
    print(f"[run] best checkpoint: {best_path}")
    result = {
        "version": cfg.version,
        "run_name": cfg.run_name,
        "select_by": cfg.select_by,
        "best_checkpoint": str(best_path),
        "best_epoch": best_record["epoch"] if best_record else None,
        "best_score": float(best_score),
        "best_metrics": _public_epoch_record(best_record),
        "final_metrics": _public_epoch_record(final_record),
        "history": history,
    }
    metrics_path = run_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[run] training metrics: {metrics_path}")
    return result
