import math
import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Cosine annealing schedule with linear warmup.

    Used across all three stages:
      Stage I: warmup_ratio=0.03, cosine decay over 3 epochs
      Stage II: warmup_ratio=0.05, cosine decay over 200 steps
      Stage III: warmup_ratio=0.03, cosine decay over 3 epochs
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        return max(min_lr_ratio, cosine_decay)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def build_optimizer_and_scheduler(
    params,
    lr: float,
    weight_decay: float,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    num_warmup_steps: int = 0,
    num_training_steps: int = 1000,
    min_lr_ratio: float = 0.0,
    optimizer_type: str = "adamw",
):
    """Unified optimizer and scheduler construction."""
    no_decay = ["bias", "layernorm", "layer_norm", "norm"]
    grouped_params = [
        {
            "params": [
                p for n, p in params
                if not any(nd in n.lower() for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in params
                if any(nd in n.lower() for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            grouped_params, lr=lr, betas=betas, eps=eps
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )

    return optimizer, scheduler