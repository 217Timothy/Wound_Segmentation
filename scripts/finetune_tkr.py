#!/usr/bin/env python3
"""Fine-tune a model on the TKR knee-wound dataset.

Same engine as the other modes, but uses the TKR data root and the gentler
TKR augmentation preset.

附帶計時功能：印硬體資訊、每 epoch 計時、結束印總結並寫 JSON。

Example:
    python scripts/finetune_tkr.py --config configs/finetune_tkr.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import finetune_tkr_defaults, load_config
from woundseg.engine import run_training
from woundseg.utils import EpochTimer, print_system_info


def main():
    sys_info = print_system_info()

    cfg = load_config(finetune_tkr_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")
    if not cfg.pretrained_ckpt:
        raise SystemExit("Fine-tuning needs 'pretrained_ckpt'. Set it in the config "
                          "or pass --pretrained_ckpt path/to/best.pt.")

    print("=" * 60)
    print(f"TKR fine-tune start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  version       : {cfg.version}")
    print(f"  run_name      : {cfg.run_name}")
    print(f"  datasets      : {cfg.datasets}")
    print(f"  epochs        : {cfg.epochs}")
    print(f"  batch         : {cfg.batch_size}")
    print(f"  pretrained    : {cfg.pretrained_ckpt}")
    print("=" * 60)

    timer = EpochTimer()
    try:
        run_training(cfg, epoch_callback=timer)
    except TypeError:
        print("[warn] run_training 不接 epoch_callback，只記總時間")
        run_training(cfg)

    print()
    print(timer.summary_text())

    out_dir = Path(cfg.out_root) / "runs" / cfg.version / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "system": sys_info,
        "config": {
            "version": cfg.version,
            "run_name": cfg.run_name,
            "datasets": cfg.datasets,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "model": cfg.model,
            "encoder_name": cfg.encoder_name,
            "pretrained_ckpt": str(cfg.pretrained_ckpt),
        },
        "summary": timer.summary(),
        "per_epoch": timer.epochs,
    }
    out_file = out_dir / "timing.json"
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[ok] 計時報告：{out_file}")


if __name__ == "__main__":
    main()
