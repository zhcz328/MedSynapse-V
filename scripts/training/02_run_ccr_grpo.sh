#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE_CONFIG="${PROJECT_ROOT}/configs/stages/stage2_ccr.yaml"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/qwen3_vl_8b.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage2_ccr"

# Stage I checkpoint (P_phi weights)
STAGE1_CKPT="${PROJECT_ROOT}/outputs/stage1_mqpm/fold_0/checkpoint_best"

NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29501}
MASTER_ADDR=${MASTER_ADDR:-localhost}

if [ ! -d "${STAGE1_CKPT}" ] && [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[ERROR] Stage I checkpoint not found: ${STAGE1_CKPT}"
    echo "[ERROR] Run Stage I first: bash scripts/training/01_run_mqpm_warmup.sh"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "  Stage II: CCR (GRPO)"
echo "  GPUs: ${NUM_GPUS} × A100-80GB"
echo "  Trainable: LoRA adapters (~83.9M params)"
echo "  LoRA: r=64, alpha=128, dropout=0.05"
echo "  GRPO: G=4, clip_eps=0.2, KL=0.02"
echo "  Reward: lambda_acc=1.0, lambda_causal=0.5"
echo "  LR: 1e-5 | Steps: 200 | Rollout BS: 32"
echo "  Temperature (sampling): 0.7"
echo "  Stage I checkpoint: ${STAGE1_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${PROJECT_ROOT}/main.py" \
    --stage 2 \
    --config "${STAGE_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --checkpoint "${STAGE1_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --wandb \
    --wandb_project medsynapse-v \
    --wandb_run stage2_ccr

echo "[INFO] Stage II training complete → ${OUTPUT_DIR}"
