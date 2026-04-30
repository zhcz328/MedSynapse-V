import os
import sys
import argparse
import logging
import yaml
import torch
import torch.distributed as dist
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("medsynapse_v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MedSynapse-V: Latent Diagnostic Memory Evolution"
    )

    # ── Mode selection ──
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--stage", type=int, choices=[1, 2, 3],
        help="Training stage: 1=MQPM, 2=CCR, 3=IMT",
    )
    mode.add_argument("--eval", action="store_true", help="Evaluation mode")
    mode.add_argument("--cache", action="store_true", help="Pre-cache MedSAM3 features")

    # ── Config paths ──
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--model_config", type=str,
        default="configs/models/qwen3_vl_8b.yaml",
    )
    parser.add_argument(
        "--encoder_config", type=str,
        default="configs/models/medsam3_frozen.yaml",
    )
    parser.add_argument("--data_config", type=str, default=None)

    # ── Checkpoints ──
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    # ── Evaluation ──
    parser.add_argument("--benchmark", type=str, nargs="+", default=None)
    parser.add_argument(
        "--eval_config", type=str,
        default="configs/datasets/eval_benchmarks.yaml",
    )

    # ── Caching ──
    parser.add_argument("--cache_output", type=str, default="cache/medsam_features")

    # ── Infrastructure ──
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="medsynapse-v")
    parser.add_argument("--wandb_run", type=str, default=None)

    # ── Hyperparameter overrides ──
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=None, help="Run single fold only")

    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def apply_overrides(config: dict, args) -> dict:
    training = config.get("training", {})
    if args.lr is not None:
        training.setdefault("optimizer", {})["lr"] = args.lr
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.max_steps is not None:
        training["training_steps"] = args.max_steps
    if args.gradient_accumulation_steps is not None:
        training["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["training"] = training
    return config


def resolve_output_dir(args, stage_config: dict) -> str:
    if args.output_dir is not None:
        return args.output_dir
    stage_name = stage_config.get("name", f"stage{args.stage}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", f"{stage_name}_{timestamp}")


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return local_rank
    return -1


def setup_wandb(args, config: dict, output_dir: str):
    if not args.wandb:
        return
    try:
        import wandb
        run_name = args.wandb_run or os.path.basename(output_dir)
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            dir=output_dir,
        )
        logger.info(f"W&B initialized: {args.wandb_project}/{run_name}")
    except ImportError:
        logger.warning("wandb not installed, skipping")


def run_training(args):
    assert args.config is not None, "--config required for training"
    stage_config = load_config(args.config)
    model_config = load_config(args.model_config)
    stage_config = apply_overrides(stage_config, args)

    output_dir = resolve_output_dir(args, stage_config)
    os.makedirs(output_dir, exist_ok=True)

    # Persist resolved config for reproducibility
    with open(os.path.join(output_dir, "config_resolved.yaml"), "w") as f:
        yaml.dump({"stage": stage_config, "model": model_config}, f)

    local_rank = setup_distributed()
    is_main = local_rank in (-1, 0)

    if is_main:
        setup_wandb(args, stage_config, output_dir)
        logger.info(f"Stage {args.stage} → {output_dir}")

    # ── Cross-validation loop (5-fold, Appendix Table 1) ──
    folds = [args.fold] if args.fold is not None else range(args.num_folds)

    for fold_idx in folds:
        fold_seed = args.seed + fold_idx
        set_seed(fold_seed)
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)

        if is_main:
            logger.info(f"=== Fold {fold_idx}/{args.num_folds - 1}, seed={fold_seed} ===")

        # Build model
        from core.builder import MedSynapseV, MedSynapseVConfig
        mconfig = MedSynapseVConfig.from_yaml(args.model_config, args.config)
        model = MedSynapseV(mconfig)
        model.build(stage=args.stage, load_encoder=(args.stage in (1, 2, 3)))

        # Resume / load prior-stage weights
        if args.checkpoint is not None:
            from utils.checkpointer import Checkpointer
            Checkpointer(fold_dir).load(model, args.checkpoint, strict=False)
            if is_main:
                logger.info(f"Loaded checkpoint: {args.checkpoint}")

        # Dispatch to engine
        if args.stage == 1:
            from engine.stage1_warmup import MQPMWarmupEngine
            engine = MQPMWarmupEngine(
                model=model, config=stage_config,
                output_dir=fold_dir, local_rank=local_rank,
            )
        elif args.stage == 2:
            from engine.stage2_rl_grpo import CCRGRPOEngine
            engine = CCRGRPOEngine(
                model=model, config=stage_config,
                output_dir=fold_dir, local_rank=local_rank,
            )
        elif args.stage == 3:
            from engine.stage3_distillation import IMTDistillationEngine
            engine = IMTDistillationEngine(
                model=model, config=stage_config,
                output_dir=fold_dir, local_rank=local_rank,
            )
        else:
            raise ValueError(f"Unknown stage: {args.stage}")

        engine.train()

        if is_main:
            logger.info(f"Fold {fold_idx} complete → {fold_dir}")

        # Free memory between folds
        del model, engine
        torch.cuda.empty_cache()

    if is_main:
        logger.info(f"All folds complete. Outputs: {output_dir}")


def run_evaluation(args):
    eval_config = load_config(args.eval_config)

    assert args.checkpoint is not None, "--checkpoint required for evaluation"

    from core.builder import MedSynapseV, MedSynapseVConfig
    from eval.evaluator import MedicalVQAEvaluator

    mconfig = MedSynapseVConfig.from_yaml(args.model_config, args.eval_config)
    model = MedSynapseV(mconfig)
    model.build(stage=0, load_encoder=False)

    from utils.checkpointer import Checkpointer
    Checkpointer(".").load(model, args.checkpoint, strict=False)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.cuda().eval()

    evaluator = MedicalVQAEvaluator(
        model=model,
        processor=model.processor,
        eval_config=eval_config,
    )

    benchmarks = args.benchmark or [
        b["name"] for b in eval_config.get("benchmarks", [])
    ]

    results = {}
    for name in benchmarks:
        logger.info(f"Evaluating: {name}")
        result = evaluator.evaluate(name)
        results[name] = result
        logger.info(f"  {name}: {result}")

    # Persist results
    out_path = os.path.join(os.path.dirname(args.checkpoint), "eval_results.yaml")
    with open(out_path, "w") as f:
        yaml.dump(results, f, default_flow_style=False)
    logger.info(f"Results → {out_path}")


def run_cache(args):
    from scripts.cache_medsam_features import cache_features

    encoder_config = load_config(args.encoder_config)
    data_cfg_path = args.data_config or "configs/datasets/stage2_rl_mixed.yaml"
    data_config = load_config(data_cfg_path)

    cache_features(
        encoder_config=encoder_config,
        data_config=data_config,
        output_dir=args.cache_output,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.stage is not None:
        run_training(args)
    elif args.eval:
        run_evaluation(args)
    elif args.cache:
        run_cache(args)


if __name__ == "__main__":
    main()
