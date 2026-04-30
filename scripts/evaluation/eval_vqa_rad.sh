#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_CONFIG="${PROJECT_ROOT}/configs/datasets/eval_benchmarks.yaml"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/qwen3_vl_8b.yaml"

# Default: use Stage III (IMT) final checkpoint
CHECKPOINT="${1:-${PROJECT_ROOT}/outputs/stage3_imt/fold_0/checkpoint_best}"

if [ ! -d "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "[USAGE] bash eval_vqa_rad.sh [checkpoint_path]"
    exit 1
fi

echo "================================================================"
echo "  Evaluation: VQA-RAD"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Decoding: greedy (temp=0, top-p=1.0)"
echo "  Max tokens: 128"
echo "  Memory: 16 diagnostic vectors (M_auto via A_psi)"
echo "================================================================"

python "${PROJECT_ROOT}/main.py" \
    --eval \
    --checkpoint "${CHECKPOINT}" \
    --model_config "${MODEL_CONFIG}" \
    --eval_config "${EVAL_CONFIG}" \
    --benchmark vqa_rad \
    --seed 42

echo "[INFO] VQA-RAD evaluation complete."
