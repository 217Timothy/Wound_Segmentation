#!/usr/bin/env python3
"""Fine-tune a model on the TKR knee-wound dataset.

Same engine as the other modes, but uses the TKR data root and the gentler
TKR augmentation preset.

Example:
    python scripts/finetune_tkr.py --config configs/finetune_tkr.yaml
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import finetune_tkr_defaults, load_config
from woundseg.engine import run_training


def main():
    cfg = load_config(finetune_tkr_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")
    if not cfg.pretrained_ckpt:
        raise SystemExit("Fine-tuning needs 'pretrained_ckpt'. Set it in the config "
                          "or pass --pretrained_ckpt path/to/best.pt.")
    run_training(cfg)


if __name__ == "__main__":
    main()
