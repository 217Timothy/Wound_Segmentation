#!/usr/bin/env python3
"""Train a wound segmentation model from scratch (general pre-training).

附帶計時功能：印硬體資訊、每 epoch 計時、結束印總結並寫 JSON。

Examples:
    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --config configs/train.yaml --epochs 120 --lr 5e-4
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import load_config, train_defaults
from woundseg.engine import run_training
from woundseg.utils import EpochTimer, print_system_info


def main():
    sys_info = print_system_info()

    cfg = load_config(train_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")

    print("=" * 60)
    print(f"Train start: {datetime.now().isoformat(timespec='seconds')}")
    print(f"  version  : {cfg.version}")
    print(f"  run_name : {cfg.run_name}")
    print(f"  datasets : {cfg.datasets}")
    print(f"  epochs   : {cfg.epochs}")
    print(f"  batch    : {cfg.batch_size}")
    print("=" * 60)

    timer = EpochTimer()
    try:
        run_training(cfg, epoch_callback=timer)
    except TypeError:
        # 舊版 run_training（沒升級到含 epoch_callback 的 loop.py）退回普通呼叫
        print("[warn] run_training 不接 epoch_callback，只記總時間")
        run_training(cfg)

    print()
    print(timer.summary_text())

    # 寫 JSON 報告
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
        },
        "summary": timer.summary(),
        "per_epoch": timer.epochs,
    }
    out_file = out_dir / "timing.json"
    out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[ok] 計時報告：{out_file}")


if __name__ == "__main__":
    main()
