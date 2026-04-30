# MedSynapse-V: Bridging Visual Perception and Clinical Intuition via Latent Memory Evolution

## Abstract

High-precision medical diagnosis relies not only on static imaging features but also on the implicit diagnostic memory experts instantly invoke during image interpretation. We pinpoint a fundamental cognitive misalignment in medical VLMs caused by discrete tokenization, leading to quantization loss, long-range information dissipation, and missing case-adaptive expertise. To bridge this gap, we propose MedSynapse-V, a framework for latent diagnostic memory evolution that simulates the experiential invocation of clinicians by dynamically synthesizing implicit diagnostic memories within the model’s hidden stream. Specifically, it begins with a Meta Query for Prior Memorization mechanism, where learnable probes retrieve structured priors from an anatomical prior encoder to generate condensed implicit memories. To ensure clinical fidelity, we introduce Causal Counterfactual Refinement (CCR) which leverages reinforcement learning and counterfactual rewards derived from region-level feature masking to quantify the causal contribution of each memory, thereby pruning redundancies and aligning latent representations with diagnostic logic. This evolutionary process culminates in Intrinsic Memory Transition (IMT), a privileged-autonomous dual-branch paradigm that internalizes teacher-branch diagnostic patterns into the student-branch via full-vocabulary divergence alignment. Comprehensive empirical evaluations across multiple datasets demonstrate that MedSynapse-V, by transferring external expertise into endogenous parameters, significantly outperforms existing state-of-the-art methods, particularly Chain-of-Thought (CoT) paradigms, in diagnostic accuracy and multi-dataset generalization without compromising the inference efficiency of standard VLMs.
## Requirements

- Python >= 3.9
- PyTorch >= 2.1.0
- CUDA >= 11.8
- Transformers >= 4.45.0
- 4× A100 80GB (recommended)
- See `requirements.txt` for full dependencies

## Quick Start

### Installation

```bash
cd MedSynapse-V
conda create -n medsynapse python=3.10
conda activate medsynapse
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

### Pre-cache MedSAM3 Features

```bash
python scripts/cache_medsam_features.py \
    --encoder_path checkpoints/medsam3_vit_b.pth \
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
bash scripts/training/03_run_imt_distill.sh
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
python main.py --eval --checkpoint outputs/stage3/final --benchmark vqa_rad
```

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
│   │   └── medsam3_frozen.yaml
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
│   ├── stage1_warmup.py
│   ├── stage2_rl_grpo.py
│   ├── stage3_distillation.py
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
│   │   └── 03_run_imt_distill.sh
│   ├── evaluation/
│   │   ├── eval_vqa_rad.sh
│   │   └── eval_omnimedvqa.sh
│   └── cache_medsam_features.py
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
