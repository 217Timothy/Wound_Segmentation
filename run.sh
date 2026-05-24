#!/usr/bin/env bash
# Convenience wrapper around the entry-point scripts.
#
# Usage:
#   ./run.sh train                              # general pre-training
#   ./run.sh finetune-wound                     # multi-class wound fine-tune
#   ./run.sh finetune-tkr                       # TKR fine-tune
#   ./run.sh evaluate                           # report per-class Dice
#   ./run.sh predict                            # batch inference
#   ./run.sh report  <version> <run_name>       # evaluate + plot curves
#
# Any extra arguments are forwarded to the underlying script, e.g.:
#   ./run.sh train --epochs 120 --lr 5e-4
set -euo pipefail

CMD="${1:-}"; shift || true

case "$CMD" in
  train)          python scripts/train.py          --config configs/train.yaml "$@" ;;
  finetune-wound) python scripts/finetune_wound.py  --config configs/finetune_wound.yaml "$@" ;;
  finetune-tkr)   python scripts/finetune_tkr.py    --config configs/finetune_tkr.yaml "$@" ;;
  evaluate)       python scripts/evaluate.py        --config configs/evaluate.yaml "$@" ;;
  predict)        python scripts/predict.py         --config configs/predict.yaml "$@" ;;
  report)
    python scripts/evaluate.py  --config configs/evaluate.yaml --version "$1" --run_name "$2"
    python scripts/visualize.py --version "$1" --run_name "$2"
    ;;
  *)
    echo "Unknown command: '${CMD}'"
    echo "Run 'cat run.sh' to see usage."
    exit 1
    ;;
esac
