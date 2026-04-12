#!/bin/bash

CONFIG_FILE="configs/run.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

VERSION=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['version'])")
RUN_NAME=$(python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['run_name'])")
DATASETS_STR=$(python -c "import yaml; print(' '.join(yaml.safe_load(open('$CONFIG_FILE'))['datasets']))")

DATASETS=($DATASETS_STR)

echo "--------------------------------------------------------"
echo " [Config Source]: ${CONFIG_FILE}"
echo ""
echo "  Version:    ${VERSION}"
echo "  Run Name:   ${RUN_NAME}"
echo "  Datasets:   ${DATASETS[@]}"
echo "--------------------------------------------------------"

set -euo pipefail
echo "========================================================"
echo "🚀 Start Pipeline: ${VERSION} / ${RUN_NAME}"
echo "========================================================"


# ========================================================
# Step 1. Plot Training Curves
# ========================================================
# echo ""
# echo "🌟 [Step 1] Visualizing Training Logs..."
# echo "--------------------------------------------------------"
# python scripts/visualize.py \
#     --version "$VERSION" \
#     --run_name "$RUN_NAME"

# ========================================================
# Step 2. Run Prediction
# ========================================================
echo ""
echo "🌟 [Step 2] Running Prediction..."
echo "--------------------------------------------------------"
python scripts/predict.py \
    --version "$VERSION" \
    --run_name "$RUN_NAME" \
    --datasets "${DATASETS[@]}"

# ========================================================
# Step 3. Run Evaluation
# ========================================================
echo ""
echo "🌟 [Step 3] Running Evaluation..."
echo "--------------------------------------------------------"
python scripts/eval.py \
    --version "$VERSION" \
    --run_name "$RUN_NAME" \
    --datasets "${DATASETS[@]}"

echo ""
echo "========================================================"
echo "✅ All Steps Completed Successfully!"
echo "========================================================"