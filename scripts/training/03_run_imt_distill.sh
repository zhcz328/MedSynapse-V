#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE_CONFIG="${PROJECT_ROOT}/configs/stages/stage3_imt.yaml"
MODEL_CONFIG="${PROJECT_ROOT}/configs/models/qwen3_vl_8b.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/stage3_imt"

# Stage II checkpoint (VLM + LoRA + P_phi)
STAGE2_CKPT="${PROJECT_ROOT}/outputs/stage2_ccr/fold_0/checkpoint_best"

NUM_GPUS=4
MASTER_PORT=${MASTER_PORT:-29502}
MASTER_ADDR=${MASTER_ADDR:-localhost}

if [ ! -d "${STAGE2_CKPT}" ] && [ ! -f "${STAGE2_CKPT}" ]; then
    echo "[ERROR] Stage II checkpoint not found: ${STAGE2_CKPT}"
    echo "[ERROR] Run Stage II first: bash scripts/training/02_run_ccr_grpo.sh"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "================================================================"
echo "  Stage III: IMT Distillation"
echo "  GPUs: ${NUM_GPUS} × A100-80GB"
echo "  Trainable: A_psi (~33.6M params)"
echo "  Divergence: JSD (beta=0.5)"
echo "  LR: 1e-4 | Epochs: 3 | BS: 32"
echo "  Teacher: P_phi + MedSAM3 (frozen)"
echo "  Student: A_psi (trainable)"
echo "  Stage II checkpoint: ${STAGE2_CKPT}"
echo "  Output: ${OUTPUT_DIR}"
echo "================================================================"

torchrun \
    --nproc_per_node=${NUM_GPUS} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    "${PROJECT_ROOT}/main.py" \
    --stage 3 \
    --config "${STAGE_CONFIG}" \
    --model_config "${MODEL_CONFIG}" \
    --checkpoint "${STAGE2_CKPT}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42 \
    --wandb \
    --wandb_project medsynapse-v \
    --wandb_run stage3_imt

echo "[INFO] Stage III training complete → ${OUTPUT_DIR}"
echo "[INFO] MedSAM3 encoder is no longer required at inference."
echo "[INFO] Inference model: VLM (8.29B) + LoRA (83.9M) + A_psi (33.6M) = 8.41B total"
