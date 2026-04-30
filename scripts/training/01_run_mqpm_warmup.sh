#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE_CONFIG="${PROJECT_ROOT}/configs/stages/stage1_mqpm.yaml"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/qwen3_vl_8b.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage1_mqpm"

NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-localhost}

# Pre-cache check
CACHE_DIR="${PROJECT_ROOT}/cache/medsam_features"
if [ ! -d "${CACHE_DIR}" ]; then
    echo "[INFO] MedSAM3 feature cache not found at ${CACHE_DIR}."
    echo "[INFO] Consider running: python scripts/cache_medsam_features.py"
fi

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "  Stage I: MQPM Warmup"
echo "  GPUs: ${NUM_GPUS} × A100-80GB"
echo "  Trainable: P_phi (~12.6M params)"
echo "  LR: 2e-4 | BS: 32 | Epochs: 3"
echo "  Mixed precision: bf16 + FlashAttention-2"
echo "  Gradient accum: 2 steps"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${PROJECT_ROOT}/main.py" \
    --stage 1 \
    --config "${STAGE_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --wandb \
    --wandb_project medsynapse-v \
    --wandb_run stage1_mqpm

echo "[INFO] Stage I training complete → ${OUTPUT_DIR}"
