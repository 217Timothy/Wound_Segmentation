"""Checkpoint saving / loading.

A checkpoint is a dict with these keys:
    epoch       int
    state_dict  model weights
    optimizer   optimizer state (optional)
    scheduler   scheduler state (optional)
    metrics     dict of validation metrics for this epoch
    model_cfg   dict describing the architecture (model/encoder/attention/...)

`model_cfg` lets inference scripts rebuild the exact architecture without
guessing from run names.
"""

import shutil
from pathlib import Path

import torch


def save_checkpoint(state: dict, is_best: bool, ckpt_dir: str | Path,
                    filename: str = "last.pt") -> None:
    """Save `state` to `ckpt_dir/filename`; also copy to best.pt when `is_best`."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / filename
    torch.save(state, last_path)
    if is_best:
        shutil.copyfile(last_path, ckpt_dir / "best.pt")


def load_checkpoint(ckpt_path: str | Path, model, optimizer=None, scheduler=None,
                    map_location: str = "cpu") -> dict:
    """Load weights (and optionally optimizer/scheduler) into the given objects."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(ckpt["state_dict"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt


def load_weights_partial(ckpt_path: str | Path, model, map_location: str = "cpu",
                         verbose: bool = True) -> dict:
    """Load only the parameters whose name and shape match the model.

    Used for transfer learning / fine-tuning where the new model may differ
    slightly from the pretrained one.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=map_location)
    src_state = ckpt.get("state_dict", ckpt)
    dst_state = model.state_dict()
    matched = {k: v for k, v in src_state.items()
               if k in dst_state and dst_state[k].shape == v.shape}
    dst_state.update(matched)
    model.load_state_dict(dst_state)
    if verbose:
        print(f"[checkpoint] loaded {len(matched)}/{len(dst_state)} "
              f"matching layers from {ckpt_path}")
    return ckpt
