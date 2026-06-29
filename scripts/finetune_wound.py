#!/usr/bin/env python3
"""Fine-tune the multi-class wound model.

Loads a pretrained checkpoint, freezes the encoder for a warm-up phase, then
trains on the wound classes. The "best" checkpoint is selected by the macro
(per-class) mean Dice, which is what the >=0.8-per-class goal tracks.

附帶計時功能：印硬體資訊、每 epoch 計時、結束印總結並寫 JSON。

Example:
    python scripts/finetune_wound.py --config configs/finetune_wound.yaml
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundseg.config import finetune_wound_defaults, load_config
from woundseg.engine import run_training
from woundseg.utils import EpochTimer, print_system_info


def main():
    started_at = datetime.now().isoformat(timespec="seconds")
    sys_info = print_system_info()

    cfg = load_config(finetune_wound_defaults())
    if not cfg.datasets:
        raise SystemExit("No datasets given. Set 'datasets' in the config or pass --datasets.")
    if not cfg.pretrained_ckpt:
        raise SystemExit("Fine-tuning needs 'pretrained_ckpt'. Set it in the config "
                          "or pass --pretrained_ckpt path/to/best.pt.")

    print("=" * 60)
    print(f"Fine-tune start: {started_at}")
    print(f"  version       : {cfg.version}")
    print(f"  run_name      : {cfg.run_name}")
    print(f"  datasets      : {cfg.datasets}")
    print(f"  epochs        : {cfg.epochs}")
    print(f"  batch         : {cfg.batch_size}")
    print(f"  pretrained    : {cfg.pretrained_ckpt}")
    print("=" * 60)

    timer = EpochTimer()
    training_result = None
    try:
        training_result = run_training(cfg, epoch_callback=timer)
    except TypeError as exc:
        if "epoch_callback" not in str(exc):
            raise
        print("[warn] run_training 不接 epoch_callback，只記總時間")
        training_result = run_training(cfg)

    print()
    print(timer.summary_text())

    if torch.cuda.is_available():
        sys_info["cuda_max_memory_allocated_GB"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3)
        sys_info["cuda_max_memory_reserved_GB"] = round(
            torch.cuda.max_memory_reserved() / 1024**3, 3)

    finished_at = datetime.now().isoformat(timespec="seconds")
    out_dir = Path(cfg.out_root) / "runs" / cfg.version / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": finished_at,
        "run": {
            "version": cfg.version,
            "run_name": cfg.run_name,
            "title": f"{cfg.version}/{cfg.run_name}",
        },
        "system": sys_info,
        "config": {
            "version": cfg.version,
            "run_name": cfg.run_name,
            "datasets": cfg.datasets,
            "data_root": cfg.data_root,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "image_size": cfg.image_size,
            "cache_data": cfg.cache_data,
            "skip_empty_masks": cfg.get("skip_empty_masks", False),
            "model": cfg.model,
            "encoder_name": cfg.encoder_name,
            "encoder_weights": cfg.encoder_weights,
            "attention": cfg.attention,
            "pretrained_ckpt": str(cfg.pretrained_ckpt),
            "freeze_encoder_epochs": cfg.freeze_encoder_epochs,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "loss_name": cfg.loss_name,
            "select_by": cfg.select_by,
            "seed": cfg.seed,
        },
        "training": training_result,
        "timing": timer.summary(),
        "summary": timer.summary(),
        "per_epoch": timer.epochs,
    }
    timing_file = out_dir / "timing.json"
    timing_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    result_file = out_dir / "results.json"
    result_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[ok] 計時報告：{timing_file}")
    print(f"[ok] 最終結果：{result_file}")


if __name__ == "__main__":
    main()
