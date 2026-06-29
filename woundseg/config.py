"""Experiment configuration.

A configuration is a plain dict with attribute access (``cfg.lr``). It is
built by layering three sources, each overriding the previous one:

    1. mode defaults  (train / finetune-wound / finetune-tkr)
    2. a YAML file     (``--config path/to/file.yaml``)
    3. command-line    (``--lr 1e-4 --epochs 80 ...``)

This keeps every entry-point script tiny: it only picks a defaults preset.
"""

import argparse
from pathlib import Path

import yaml


class Config(dict):
    """A dict whose keys are also accessible as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


# --------------------------------------------------------------------------
# Defaults shared by every training mode.
# --------------------------------------------------------------------------
BASE_DEFAULTS = {
    # experiment identity (used to build output paths)
    "version": "exp",
    "run_name": "run1",

    # data
    "data_root": "data/processed/wound",
    "datasets": [],
    "image_size": 512,
    "cache_data": False,
    "skip_empty_masks": False,      # drop samples whose mask has no foreground
    "augmentation": "default",      # "default" | "tkr"

    # model
    "model": "efficientunet",       # "unet" | "resunet" | "efficientunet"
    "encoder_name": "efficientnet-b3",
    "encoder_weights": "imagenet",  # set to null when fine-tuning from a ckpt
    "attention": "scse",            # "scse" | null

    # optimization
    "epochs": 100,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 8,
    "num_workers": 4,
    "grad_clip": 1.0,
    "loss_name": "focal_tversky",   # focal_tversky | bce_tversky | bce_dice | dice
    "amp": True,
    "seed": 42,

    # transfer learning
    "pretrained_ckpt": None,
    "freeze_encoder_epochs": 0,
    "resume": True,                 # resume from last.pt if present

    # checkpoint selection / output
    "select_by": "dice",            # "dice" (micro) | "mean_dice" (per-class macro)
    "out_root": "outputs",
}


def _preset(**overrides) -> dict:
    cfg = dict(BASE_DEFAULTS)
    cfg.update(overrides)
    return cfg


def train_defaults() -> dict:
    """General pre-training on the large segmentation datasets."""
    return _preset(
        select_by="dice",
        resume=True,
    )


def finetune_wound_defaults() -> dict:
    """Fine-tuning the multi-class wound model (the >=0.8 Dice-per-class goal)."""
    return _preset(
        encoder_weights=None,
        epochs=50,
        lr=1e-5,
        batch_size=4,
        freeze_encoder_epochs=10,
        select_by="mean_dice",
        resume=False,
    )


def finetune_tkr_defaults() -> dict:
    """Fine-tuning on the TKR knee-wound dataset."""
    return _preset(
        data_root="data/processed/tkr",
        datasets=["TKR_Knee"],
        encoder_weights=None,
        augmentation="tkr",
        epochs=50,
        lr=5e-5,
        batch_size=8,
        loss_name="bce_dice",
        freeze_encoder_epochs=2,
        select_by="dice",
        resume=False,
    )


def load_config(defaults: dict, argv: list[str] | None = None) -> Config:
    """Build a :class:`Config` from `defaults` + optional YAML + CLI overrides."""
    # Stage 1: pull out --config first so the YAML can act as defaults.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None,
                     help="Path to a YAML config file.")
    known, rest = pre.parse_known_args(argv)

    cfg = Config(defaults)
    if known.config:
        path = Path(known.config)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            cfg.update(yaml.safe_load(f) or {})
        print(f"[config] loaded {path}")

    # Stage 2: build a parser exposing every key as an optional CLI override.
    parser = argparse.ArgumentParser(
        parents=[pre], description="Override any config value from the CLI.")
    for key, value in defaults.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            # paired flags so booleans can be forced either way
            parser.add_argument(flag, dest=key, action="store_true", default=None)
            parser.add_argument(f"--no-{key}", dest=key, action="store_false", default=None)
        elif isinstance(value, list):
            parser.add_argument(flag, dest=key, nargs="+", default=None)
        else:
            argtype = type(value) if value is not None else str
            parser.add_argument(flag, dest=key, type=argtype, default=None)
    args = parser.parse_args(rest)

    # Stage 3: apply CLI values that were actually provided.
    for key, value in vars(args).items():
        if key != "config" and value is not None:
            cfg[key] = value
    return cfg


def save_config(cfg: Config, path: str | Path) -> None:
    """Dump the resolved config next to the run outputs for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=True, allow_unicode=True)
