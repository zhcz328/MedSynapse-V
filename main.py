import argparse
import logging
import os
from datetime import datetime

import torch
import torch.distributed as dist
import yaml

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
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage", type=int, choices=[1, 2, 3])
    mode.add_argument("--eval", action="store_true")
    mode.add_argument("--cache", action="store_true")

    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model_config", type=str, default="configs/models/qwen3_vl_8b.yaml")
    parser.add_argument("--encoder_config", type=str, default="configs/models/medsam_encoder_frozen.yaml")
    parser.add_argument("--data_config", type=str, default=None)
    parser.add_argument("--eval_config", type=str, default="configs/datasets/eval_benchmarks.yaml")

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--benchmark", type=str, nargs="+", default=None)
    parser.add_argument("--cache_output", type=str, default="cache/medsam_features")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="medsynapse-v")
    parser.add_argument("--wandb_run", type=str, default=None)

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_folds", type=int, default=1)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def apply_overrides(config: dict, args) -> dict:
    training = config.get("training", {})
    if args.lr is not None:
        training.setdefault("optimizer", {})["lr"] = args.lr
    if args.batch_size is not None:
        training["batch_size"] = args.batch_size
        training["rollout_batch_size"] = args.batch_size
    if args.epochs is not None:
        training["epochs"] = args.epochs
    if args.max_steps is not None:
        training["training_steps"] = args.max_steps
    if args.gradient_accumulation_steps is not None:
        training["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    config["training"] = training
    return config


def resolve_output_dir(args, stage_config: dict) -> str:
    if args.output_dir:
        return args.output_dir
    stage_name = stage_config.get("name", f"stage{args.stage}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", f"{stage_name}_{timestamp}")


def setup_distributed():
    if "LOCAL_RANK" not in os.environ:
        return -1
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return local_rank


def get_device(args, local_rank: int):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{max(local_rank, 0)}")
    return torch.device("cpu")


def setup_wandb(args, config: dict, output_dir: str):
    if not args.wandb:
        return
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping")
        return
    run_name = args.wandb_run or os.path.basename(output_dir)
    wandb.init(project=args.wandb_project, name=run_name, config=config, dir=output_dir)


def run_training(args):
    if args.config is None:
        raise ValueError("--config required for training")

    stage_config = apply_overrides(load_config(args.config), args)
    output_dir = resolve_output_dir(args, stage_config)
    os.makedirs(output_dir, exist_ok=True)

    local_rank = setup_distributed()
    is_main = local_rank in (-1, 0)
    device = get_device(args, local_rank)

    if is_main:
        setup_wandb(args, stage_config, output_dir)
        logger.info("Stage %s output: %s", args.stage, output_dir)
        with open(os.path.join(output_dir, "config_resolved.yaml"), "w") as f:
            yaml.safe_dump(
                {
                    "stage": stage_config,
                    "model": load_config(args.model_config),
                    "encoder": load_config(args.encoder_config),
                },
                f,
                sort_keys=False,
            )

    folds = [args.fold] if args.fold is not None else range(args.num_folds)
    for fold_idx in folds:
        set_seed(args.seed + fold_idx)
        fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        if is_main:
            logger.info("Fold %s/%s on %s", fold_idx, args.num_folds - 1, device)

        from core.builder import MedSynapseV, MedSynapseVConfig

        mconfig = MedSynapseVConfig.from_yaml(
            args.model_config,
            args.config,
            args.encoder_config,
        )
        model = MedSynapseV(mconfig).build(stage=args.stage, load_encoder=True)
        model.to(device)

        if args.checkpoint:
            from utils.checkpointer import Checkpointer

            Checkpointer(fold_dir).load(model, args.checkpoint, strict=False)

        data_cfg_path = args.data_config or stage_config.get("data", {}).get("config")
        data_config = load_config(data_cfg_path) if data_cfg_path else {}
        from data.loader import build_stage_dataloaders

        train_loader, eval_loader = build_stage_dataloaders(
            stage_config=stage_config,
            data_config=data_config,
            processor=model.processor,
            stage=args.stage,
            seed=args.seed,
            fold_idx=fold_idx,
        )

        if args.stage == 1:
            from engine.stage1_mqpm_warmup import Stage1WarmupEngine

            engine = Stage1WarmupEngine(
                model=model,
                processor=model.processor,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                config=stage_config,
                output_dir=fold_dir,
                device=device,
                local_rank=local_rank,
            )
        elif args.stage == 2:
            from engine.stage2_ccr_grpo import Stage2GRPOEngine

            engine = Stage2GRPOEngine(
                model=model,
                processor=model.processor,
                train_dataloader=train_loader,
                config=stage_config,
                output_dir=fold_dir,
                device=device,
                local_rank=local_rank,
            )
        else:
            from engine.stage3_imt_distillation import Stage3DistillationEngine

            engine = Stage3DistillationEngine(
                model=model,
                processor=model.processor,
                train_dataloader=train_loader,
                eval_dataloader=eval_loader,
                config=stage_config,
                output_dir=fold_dir,
                device=device,
                local_rank=local_rank,
            )

        engine.train()
        del model, engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if is_main:
        logger.info("Training complete: %s", output_dir)


def run_evaluation(args):
    if args.checkpoint is None:
        raise ValueError("--checkpoint required for evaluation")

    set_seed(args.seed)
    eval_config = load_config(args.eval_config)
    device = get_device(args, -1)

    from core.builder import MedSynapseV, MedSynapseVConfig
    from data.loader import build_eval_loaders
    from eval.evaluator import Evaluator
    from utils.checkpointer import Checkpointer

    lora_config = args.config
    if lora_config is None and os.path.exists("configs/stages/stage2_ccr.yaml"):
        lora_config = "configs/stages/stage2_ccr.yaml"
    mconfig = MedSynapseVConfig.from_yaml(args.model_config, lora_config, args.encoder_config)
    model = MedSynapseV(mconfig).build(stage=0, load_encoder=False)
    Checkpointer(".").load(model, args.checkpoint, strict=False)
    model.to(device).eval()

    loaders, bench_cfgs = build_eval_loaders(
        eval_config,
        processor=model.processor,
        benchmarks=args.benchmark,
    )
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "eval")
    evaluator = Evaluator(
        model=model,
        benchmark_config=eval_config,
        output_dir=output_dir,
        max_new_tokens=eval_config.get("evaluation", {}).get("decoding", {}).get("max_new_tokens", 128),
        temperature=eval_config.get("evaluation", {}).get("decoding", {}).get("temperature", 0.0),
        top_p=eval_config.get("evaluation", {}).get("decoding", {}).get("top_p", 1.0),
        use_imt=True,
    )
    summary = evaluator.evaluate_all(loaders, bench_cfgs)
    out_path = os.path.join(output_dir, "eval_results.yaml")
    with open(out_path, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)
    logger.info("Evaluation results: %s", out_path)


def run_cache(args):
    from scripts.cache_anatomical_priors import cache_features

    data_cfg_path = args.data_config or "configs/datasets/stage2_rl_mixed.yaml"
    cache_features(
        encoder_config=load_config(args.encoder_config),
        data_config=load_config(data_cfg_path),
        output_dir=args.cache_output,
        model_config=load_config(args.model_config),
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
