#!/usr/bin/env python3
"""Plot training curves from a run's CSV log.

Reads outputs/logs/<version>/<run_name>.csv and saves a curves.png next to the
run outputs.

Example:
    python scripts/visualize.py --version v3 --run_name run1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--out_root", default="outputs")
    return parser.parse_args()


def _annotate_extreme(ax, x, y, kind="max"):
    """Mark the best point on a curve."""
    idx = y.idxmax() if kind == "max" else y.idxmin()
    ax.scatter(x[idx], y[idx], s=110, color="crimson", zorder=5,
               edgecolors="white", linewidth=2)
    ax.annotate(f"{kind}: {y[idx]:.4f}\n(epoch {int(x[idx])})",
                xy=(x[idx], y[idx]), fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))


def main():
    args = get_args()
    out_root = Path(args.out_root)
    log_path = out_root / "logs" / args.version / f"{args.run_name}.csv"
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    df = pd.read_csv(log_path)
    epochs = df["epoch"]

    panels = [
        ("val_dice", "Validation Dice (micro)", "max"),
        ("val_mean_dice", "Validation Mean Dice (per-class)", "max"),
        ("val_iou", "Validation IoU", "max"),
        ("val_loss", "Validation Loss", "min"),
    ]
    panels = [p for p in panels if p[0] in df.columns]

    plt.style.use("ggplot")
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(7 * (len(panels) + 1), 5))

    for ax, (col, title, kind) in zip(axes, panels):
        ax.plot(epochs, df[col], marker="o", markersize=3, color="royalblue")
        _annotate_extreme(ax, epochs, df[col], kind)
        ax.set_title(f"{title}\n({args.version}/{args.run_name})")
        ax.set_xlabel("epoch")

    loss_ax = axes[-1]
    if "train_loss" in df.columns:
        loss_ax.plot(epochs, df["train_loss"], marker="o", markersize=3,
                     label="train", color="darkorange")
    if "val_loss" in df.columns:
        loss_ax.plot(epochs, df["val_loss"], marker="o", markersize=3,
                     label="val", color="royalblue")
    loss_ax.set_title("Train vs Val Loss")
    loss_ax.set_xlabel("epoch")
    loss_ax.legend()

    fig.tight_layout()
    out_dir = out_root / "runs" / args.version / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "curves.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[visualize] saved {out_path}")


if __name__ == "__main__":
    main()
