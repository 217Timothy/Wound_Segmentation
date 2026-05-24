#!/usr/bin/env python3
"""Train a wound segmentation model from scratch (general pre-training).

Examples:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --epochs 120 --lr 5e-4
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import load_config, train_defaults
from woundseg.engine import run_training


def main():
    cfg = load_config(train_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")
    run_training(cfg)


if __name__ == "__main__":
    main()
