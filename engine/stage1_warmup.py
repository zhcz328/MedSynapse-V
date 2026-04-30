import os
import time
import math
import logging
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from tqdm import tqdm

from losses.ntp_loss import NTPLoss
from engine.lr_scheduler import build_optimizer_and_scheduler
from utils.checkpointer import CheckpointManager

logger = logging.getLogger(__name__)


class Stage1WarmupEngine:
    """
    Training engine for Stage I: MQPM semantic alignment warmup.

    Performs next-token prediction with diagnostic implicit memory M
    injected into the VLM hidden stream. Only P_phi receives gradients;
    the VLM backbone theta and anatomical encoder E_ana remain frozen.
    """

    def __init__(
        self,
        model,
        processor,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        output_dir: str = "outputs/stage1",
        device: torch.device = None,
        local_rank: int = -1,
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.output_dir = output_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_rank = local_rank

        os.makedirs(output_dir, exist_ok=True)

        train_cfg = self.config.get("training", {})
        self.epochs = train_cfg.get("epochs", 3)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 2)
        self.use_amp = train_cfg.get("mixed_precision", "bf16") != "fp32"
        self.amp_dtype = torch.bfloat16 if train_cfg.get("mixed_precision") == "bf16" else torch.float16

        log_cfg = self.config.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 50)
        self.save_interval = log_cfg.get("save_interval", 500)
        self.eval_interval = log_cfg.get("eval_interval", 500)

        self.loss_fn = NTPLoss(
            label_smoothing=self.config.get("loss", {}).get("label_smoothing", 0.0)
        )

        self._setup_optimizer()
        self.checkpointer = CheckpointManager(output_dir)
        self.global_step = 0

    def _setup_optimizer(self):
        train_cfg = self.config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        sched_cfg = train_cfg.get("scheduler", {})

        trainable_params = [
            (n, p) for n, p in self.model.memory_sampler.named_parameters()
            if p.requires_grad
        ]
        total_trainable = sum(p.numel() for _, p in trainable_params)
        logger.info(f"Stage I trainable parameters: {total_trainable / 1e6:.2f}M")

        num_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        total_steps = num_steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * sched_cfg.get("warmup_ratio", 0.03))

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            params=trainable_params,
            lr=opt_cfg.get("lr", 2e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=sched_cfg.get("min_lr_ratio", 0.0),
        )

        self.scaler = GradScaler(enabled=(self.amp_dtype == torch.float16))

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Single forward pass with memory injection and NTP loss."""
        images = batch.get("pixel_values")
        if images is not None:
            images = images.to(self.device)

        # Extract anatomical features and synthesize memory
        with torch.no_grad():
            enc_output = self.model.encoder(images)
            encoder_features = enc_output["features"]

        # Memory synthesis (gradients flow through P_phi)
        memory = self.model.memory_sampler(encoder_features)

        # Get input embeddings from VLM
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels", input_ids.clone()).to(self.device)

        with torch.no_grad():
            inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)

        # Inject memory into hidden stream
        injected = self.model.injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory,
            attention_mask=attention_mask,
        )

        # Forward through VLM
        outputs = self.model.vlm(
            inputs_embeds=injected["inputs_embeds"],
            attention_mask=injected["attention_mask"],
            return_dict=True,
        )

        # Compute NTP loss; pad labels to match injected sequence length
        N = memory.shape[1]
        B, L = labels.shape
        padded_labels = torch.full(
            (B, L + N), -100, dtype=labels.dtype, device=labels.device
        )
        # Memory positions are masked from loss
        injection_pos = attention_mask.sum(dim=1).long()
        for i in range(B):
            pos = injection_pos[i].item()
            padded_labels[i, :pos] = labels[i, :pos]
            padded_labels[i, pos + N:pos + N + (L - pos)] = labels[i, pos:]

        loss = self.loss_fn(outputs.logits, padded_labels)

        with torch.no_grad():
            token_acc = self.loss_fn.compute_token_accuracy(outputs.logits, padded_labels)

        return {"loss": loss, "token_accuracy": token_acc}

    def train(self):
        logger.info(f"Starting Stage I warmup: {self.epochs} epochs")
        self.model.memory_sampler.train()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            progress = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=(self.local_rank > 0),
            )

            self.optimizer.zero_grad()

            for step, batch in enumerate(progress):
                with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                    result = self._forward_step(batch)
                    loss = result["loss"] / self.gradient_accumulation_steps

                if self.amp_dtype == torch.float16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    if self.amp_dtype == torch.float16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.memory_sampler.parameters(),
                        self.max_grad_norm,
                    )
                    if self.amp_dtype == torch.float16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    epoch_loss += result["loss"].item()
                    epoch_steps += 1

                    if self.global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        lr = self.scheduler.get_last_lr()[0]
                        logger.info(
                            f"Step {self.global_step} | Loss: {avg_loss:.4f} | "
                            f"Acc: {result['token_accuracy']:.4f} | LR: {lr:.2e}"
                        )
                        progress.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                    if self.global_step % self.save_interval == 0:
                        self.checkpointer.save(
                            {"memory_sampler": self.model.memory_sampler.state_dict()},
                            self.global_step, tag="stage1",
                        )

                    if self.eval_dataloader and self.global_step % self.eval_interval == 0:
                        self._evaluate()

            logger.info(
                f"Epoch {epoch + 1} complete | Avg Loss: {epoch_loss / max(epoch_steps, 1):.4f}"
            )

        # Save final checkpoint
        self.checkpointer.save(
            {"memory_sampler": self.model.memory_sampler.state_dict()},
            self.global_step, tag="stage1_final",
        )
        logger.info("Stage I warmup complete")

    @torch.no_grad()
    def _evaluate(self):
        self.model.memory_sampler.eval()
        total_loss, total_acc, num_batches = 0.0, 0.0, 0

        for batch in self.eval_dataloader:
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                result = self._forward_step(batch)
            total_loss += result["loss"].item()
            total_acc += result["token_accuracy"]
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        avg_acc = total_acc / max(num_batches, 1)
        logger.info(f"Eval @ step {self.global_step} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        self.model.memory_sampler.train()