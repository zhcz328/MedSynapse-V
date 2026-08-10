> 🌟 Accepted to ECCV 2026 

# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

## Abstract

High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model’s hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets show that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference efficiency of standard VLMs.
## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA >= 11.8
- Transformers >= 4.45.0
- 4× A100 80GB (recommended)
- See `requirements.txt` for full dependencies

## Quick Start

### What This Code Implements

This repository follows the three-stage MedSynapse-V training recipe from the paper:

1. **Stage I - MQPM warmup**: a frozen anatomical encoder extracts spatial prior features; learnable meta-query probes condense them into implicit diagnostic memory tokens that are injected into the VLM hidden stream.
2. **Stage II - CCR**: the memory sampler is frozen and the policy is optimized with GRPO-style rollouts plus an accuracy reward and a counterfactual reward from masked anatomical features.
3. **Stage III - IMT**: a privileged branch using the anatomical encoder teaches an autonomous memory module through full-vocabulary JSD alignment, so inference can run without the anatomical encoder.

This code path expects real Qwen3-VL weights, a MedSAM checkpoint, and the medical VQA datasets declared in `configs/datasets/*.yaml`. It does not include fallback miniature models or generated substitute data.

### Reproducibility Checklist

This repository can reproduce the MedSynapse-V training pipeline only when the following external assets are provided:

- Qwen/Qwen3-VL-8B-Instruct weights, or a compatible Qwen-VL checkpoint matching `configs/models/qwen3_vl_8b.yaml`.
- A real MedSAM/SAM ViT-B checkpoint at `checkpoints/medsam_vit_b.pth`. The encoder wrapper now prefers the open-source `segment_anything` image encoder so MedSAM checkpoints with `image_encoder.*` keys are loaded correctly.
- Access to the configured medical VQA datasets, including any Hugging Face gated datasets and local image roots required by SLAKE-like records.
- The same three-stage checkpoint chain: Stage I `memory_sampler` -> Stage II LoRA + `memory_sampler` -> Stage III LoRA + `autonomous_module`.
- Matching evaluation splits and seeds. The paper-scale protocol uses `--num_folds 5`; the default is one fold for development.

Without the exact released checkpoints, dataset revisions, and paper evaluation splits, the code can reproduce the method but should not be expected to reproduce the paper table numbers bit-for-bit.

### Installation

```bash
cd MedSynapse-V
conda create -n medsynapse python=3.10
conda activate medsynapse
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Cache Anatomical Priors

```bash
python scripts/cache_anatomical_priors.py \
    --encoder_path checkpoints/medsam_vit_b.pth \
    --data_config configs/datasets/stage2_rl_mixed.yaml \
    --output_dir cache/medsam_features
```

### Training

```bash
# Stage I: Meta Query for Prior Memorization
bash scripts/training/01_run_mqpm_warmup.sh

# Stage II: Causal Counterfactual Refinement
bash scripts/training/02_run_ccr_grpo.sh

# Stage III: Intrinsic Memory Transition
bash scripts/training/03_run_imt_distillation.sh
```

### Evaluation

```bash
bash scripts/evaluation/eval_vqa_rad.sh
bash scripts/evaluation/eval_omnimedvqa.sh
```

### Unified CLI

```bash
python main.py --stage 1 --config configs/stages/stage1_mqpm.yaml
python main.py --stage 2 --config configs/stages/stage2_ccr.yaml
python main.py --stage 3 --config configs/stages/stage3_imt.yaml
python main.py --eval --checkpoint outputs/stage3_imt/fold_0/checkpoint_best.pt --benchmark vqa_rad
```

For paper-scale runs, pass `--num_folds 5` if you want the five-seed protocol described in the manuscript. The default is `--num_folds 1` so a single command does not accidentally launch five full runs.

## Project Structure

```
MedSynapse-V/
├── configs/
│   ├── datasets/
│   │   ├── stage1_pubmedvision.yaml
│   │   ├── stage2_rl_mixed.yaml
│   │   └── eval_benchmarks.yaml
│   ├── models/
│   │   ├── qwen3_vl_8b.yaml
│   │   └── medsam_encoder_frozen.yaml
│   └── stages/
│       ├── stage1_mqpm.yaml
│       ├── stage2_ccr.yaml
│       └── stage3_imt.yaml
│
├── core/
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── diagnostic_sampler.py
│   │   ├── autonomous_module.py
│   │   └── hidden_injector.py
│   ├── encoders/
│   │   ├── medsam_wrapper.py
│   │   └── qwen_vision.py
│   └── builder.py
│
├── data/
│   ├── datasets/
│   │   ├── omnimedvqa.py
│   │   ├── slake_pathvqa.py
│   │   └── gmai_mmbench.py
│   ├── loader.py
│   └── templates.py
│
├── engine/
│   ├── stage1_mqpm_warmup.py
│   ├── stage2_ccr_grpo.py
│   ├── stage3_imt_distillation.py
│   └── lr_scheduler.py
│
├── rewards/
│   ├── __init__.py
│   ├── accuracy_reward.py
│   └── causal_reward.py
│
├── losses/
│   ├── __init__.py
│   ├── ntp_loss.py
│   ├── grpo_loss.py
│   └── jsd_loss.py
│
├── eval/
│   ├── evaluator.py
│   ├── regex_extractor.py
│   └── metrics.py
│
├── scripts/
│   ├── training/
│   │   ├── 01_run_mqpm_warmup.sh
│   │   ├── 02_run_ccr_grpo.sh
│   │   └── 03_run_imt_distillation.sh
│   ├── evaluation/
│   │   ├── eval_vqa_rad.sh
│   │   └── eval_omnimedvqa.sh
│   └── cache_anatomical_priors.py
│
├── utils/
│   ├── visualization.py
│   └── checkpointer.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Hardware Requirements

| Stage | GPUs | Time | Peak Memory |
|-------|------|------|-------------|
| Stage I (MQPM) | 4× A100 80GB | ~8h | ~22 GB/GPU |
| Stage II (CCR) | 4× A100 80GB | ~18h | ~45 GB/GPU |
| Stage III (IMT) | 4× A100 80GB | ~12h | ~30 GB/GPU |
| **Total** | | **~38h** | |


## License

This project is released under the Apache 2.0 License.
