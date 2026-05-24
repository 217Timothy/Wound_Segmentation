#!/usr/bin/env python3
"""Fine-tune the multi-class wound model.

Loads a pretrained checkpoint, freezes the encoder for a warm-up phase, then
trains on the wound classes. The "best" checkpoint is selected by the macro
(per-class) mean Dice, which is what the >=0.8-per-class goal tracks.

Example:
    python scripts/finetune_wound.py --config configs/finetune_wound.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import finetune_wound_defaults, load_config
from woundseg.engine import run_training


def main():
    cfg = load_config(finetune_wound_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")
    if not cfg.pretrained_ckpt:
        raise SystemExit("Fine-tuning needs 'pretrained_ckpt'. Set it in the config "
                          "or pass --pretrained_ckpt path/to/best.pt.")
    run_training(cfg)


if __name__ == "__main__":
    main()
