#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EVAL_CONFIG="${PROJECT_ROOT}/configs/datasets/eval_benchmarks.yaml"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/qwen3_vl_8b.yaml"

CHECKPOINT="${1:-${PROJECT_ROOT}/outputs/stage3_imt/fold_0/checkpoint_best}"

if [ ! -d "${CHECKPOINT}" ] && [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "[USAGE] bash eval_omnimedvqa.sh [checkpoint_path]"
    exit 1
fi

echo "================================================================"
echo "  Evaluation: OmniMedVQA (8 modalities)"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Decoding: greedy (temp=0, top-p=1.0)"
echo "  Max tokens: 128"
echo "  Reports: overall accuracy + per-modality breakdown"
echo "================================================================"

python "${PROJECT_ROOT}/main.py" \
    --eval \
    --checkpoint "${CHECKPOINT}" \
    --model_config "${MODEL_CONFIG}" \
    --eval_config "${EVAL_CONFIG}" \
    --benchmark omnimedvqa \
    --seed 42

echo "[INFO] OmniMedVQA evaluation complete."

# ── Run full benchmark suite ──
echo ""
echo "To run the complete evaluation suite, use:"
echo "  python main.py --eval --checkpoint ${CHECKPOINT} \\"
echo "      --benchmark vqa_rad slake pathvqa pmc_vqa mmmu_health medxpertqa_mm gmai_mmbench"
