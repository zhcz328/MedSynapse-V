import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

from losses.jsd_loss import JSDLoss
from engine.lr_scheduler import build_optimizer_and_scheduler
from utils.checkpointer import CheckpointManager

logger = logging.getLogger(__name__)


class Stage3DistillationEngine:
    """
    Training engine for Stage III: Intrinsic Memory Transition.

    Dual-branch forward pass aligns the student's next-token distribution
    (conditioned on M_auto from A_psi) with the teacher's distribution
    (conditioned on M_pri from E_ana + P_phi) using generalized JSD.
    """

    def __init__(
        self,
        model,
        processor,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        config: Optional[Dict] = None,
        output_dir: str = "outputs/stage3",
        device: torch.device = None,
        local_rank: int = -1,
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.output_dir = output_dir
        self.device = device or torch.device("cuda")
        self.local_rank = local_rank

        os.makedirs(output_dir, exist_ok=True)

        dist_cfg = self.config.get("distillation", {})
        self.jsd_beta = dist_cfg.get("jsd_beta", 0.5)
        self.temperature = dist_cfg.get("temperature", 1.0)
        self.on_policy_samples = dist_cfg.get("on_policy_samples", 1)
        self.max_gen_len = dist_cfg.get("max_generation_length", 512)

        train_cfg = self.config.get("training", {})
        self.epochs = train_cfg.get("epochs", 3)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 2)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.use_amp = train_cfg.get("mixed_precision", "bf16") != "fp32"
        self.amp_dtype = torch.bfloat16

        log_cfg = self.config.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 50)
        self.save_interval = log_cfg.get("save_interval", 500)

        self.jsd_loss = JSDLoss(beta=self.jsd_beta, temperature=self.temperature)
        self._setup_optimizer()
        self.checkpointer = CheckpointManager(output_dir)
        self.global_step = 0

    def _setup_optimizer(self):
        train_cfg = self.config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        sched_cfg = train_cfg.get("scheduler", {})

        trainable_params = [
            (n, p) for n, p in self.model.autonomous_module.named_parameters()
            if p.requires_grad
        ]
        total = sum(p.numel() for _, p in trainable_params)
        logger.info(f"Stage III trainable parameters (A_psi): {total / 1e6:.2f}M")

        num_steps_per_epoch = len(self.train_dataloader) // self.gradient_accumulation_steps
        total_steps = num_steps_per_epoch * self.epochs
        warmup_steps = int(total_steps * sched_cfg.get("warmup_ratio", 0.03))

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            params=trainable_params,
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    def _get_teacher_logits(
        self,
        batch: Dict,
        memory_pri: torch.Tensor,
        trajectory_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Teacher branch: compute pi^+(· | X, q, M_pri, y_hat_{<n}).
        All parameters frozen; no gradients.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        with torch.no_grad():
            inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)
            injected = self.model.injector.inject(
                inputs_embeds=inputs_embeds,
                memory=memory_pri,
                attention_mask=attention_mask,
            )

            # Append trajectory tokens
            traj_embeds = self.model.vlm.get_input_embeddings()(trajectory_ids)
            full_embeds = torch.cat([injected["inputs_embeds"], traj_embeds], dim=1)
            full_mask = torch.cat([
                injected["attention_mask"],
                torch.ones(
                    trajectory_ids.shape, dtype=injected["attention_mask"].dtype,
                    device=self.device,
                ),
            ], dim=1)

            outputs = self.model.vlm(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                return_dict=True,
            )

        # Return logits over the trajectory portion
        context_len = injected["inputs_embeds"].shape[1]
        return outputs.logits[:, context_len - 1:-1, :]

    def _get_student_logits(
        self,
        batch: Dict,
        memory_auto: torch.Tensor,
        trajectory_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Student branch: compute pi^-(· | X, q, M_auto, y_hat_{<n}).
        Gradients flow through A_psi via memory_auto.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # VLM embeddings (no grad for the embedding layer itself)
        with torch.no_grad():
            inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)

        # Inject autonomous memory (grad enabled for A_psi)
        injected = self.model.injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory_auto,
            attention_mask=attention_mask,
        )

        traj_embeds = self.model.vlm.get_input_embeddings()(trajectory_ids)
        full_embeds = torch.cat([injected["inputs_embeds"], traj_embeds.detach()], dim=1)
        full_mask = torch.cat([
            injected["attention_mask"],
            torch.ones(
                trajectory_ids.shape, dtype=injected["attention_mask"].dtype,
                device=self.device,
            ),
        ], dim=1)

        outputs = self.model.vlm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            return_dict=True,
        )

        context_len = injected["inputs_embeds"].shape[1]
        return outputs.logits[:, context_len - 1:-1, :]

    @torch.no_grad()
    def _sample_on_policy_trajectory(
        self,
        batch: Dict,
        memory_auto: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample on-policy trajectory y_hat ~ pi^-(· | X, q, M_auto).
        Used as the shared prefix for both teacher and student branches.
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)
        injected = self.model.injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory_auto.detach(),
            attention_mask=attention_mask,
        )

        outputs = self.model.vlm.generate(
            inputs_embeds=injected["inputs_embeds"],
            attention_mask=injected["attention_mask"],
            max_new_tokens=self.max_gen_len,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.95,
            return_dict_in_generate=True,
        )

        # Extract generated portion only
        gen_ids = outputs.sequences[:, injected["inputs_embeds"].shape[1]:]
        return gen_ids

    def _forward_step(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Single distillation step: dual-branch JSD alignment."""
        images = batch.get("pixel_values")
        if images is not None:
            images = images.to(self.device)

        # Teacher: privileged memory from E_ana + P_phi (frozen)
        with torch.no_grad():
            mem_output = self.model.generate_memory_privileged(images)
            memory_pri = mem_output["memory"]

        # Student: autonomous memory from A_psi (trainable)
        input_ids = batch["input_ids"].to(self.device)
        memory_auto = self.model.generate_memory_autonomous(
            input_ids=input_ids,
            pixel_values=batch.get("pixel_values", None),
            image_grid_thw=batch.get("image_grid_thw", None),
            attention_mask=batch.get("attention_mask", None),
        )

        # Sample on-policy trajectory from student
        trajectory_ids = self._sample_on_policy_trajectory(batch, memory_auto)

        # Get logits from both branches
        with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
            teacher_logits = self._get_teacher_logits(batch, memory_pri, trajectory_ids)
            student_logits = self._get_student_logits(batch, memory_auto, trajectory_ids)

            # Mask for valid trajectory positions
            seq_mask = (trajectory_ids != self.processor.tokenizer.pad_token_id).float()

            # Compute JSD loss (Eq. 7-8)
            loss_dict = self.jsd_loss(
                teacher_logits=teacher_logits,
                student_logits=student_logits,
                sequence_mask=seq_mask,
            )

        return loss_dict

    def train(self):
        logger.info(f"Starting Stage III IMT distillation: {self.epochs} epochs")
        self.model.autonomous_module.train()
        # Freeze everything except A_psi
        self.model.vlm.eval()
        if self.model.memory_sampler is not None:
            self.model.memory_sampler.eval()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_agreement = 0.0
            epoch_steps = 0

            progress = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                disable=(self.local_rank > 0),
            )

            self.optimizer.zero_grad()

            for step, batch in enumerate(progress):
                result = self._forward_step(batch)
                loss = result["loss"] / self.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.autonomous_module.parameters(),
                        self.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    epoch_loss += result["loss"].item()
                    epoch_agreement += result["output_agreement"].item()
                    epoch_steps += 1

                    if self.global_step % self.log_interval == 0:
                        avg_loss = epoch_loss / max(epoch_steps, 1)
                        avg_agree = epoch_agreement / max(epoch_steps, 1)
                        logger.info(
                            f"Step {self.global_step} | JSD: {avg_loss:.4f} | "
                            f"Agreement: {avg_agree:.4f} | "
                            f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                        )
                        progress.set_postfix(jsd=f"{avg_loss:.4f}", agree=f"{avg_agree:.3f}")

                    if self.global_step % self.save_interval == 0:
                        self.checkpointer.save(
                            {"autonomous_module": self.model.autonomous_module.state_dict()},
                            self.global_step, tag="stage3",
                        )

            logger.info(
                f"Epoch {epoch + 1} | JSD: {epoch_loss / max(epoch_steps, 1):.4f} | "
                f"Agreement: {epoch_agreement / max(epoch_steps, 1):.4f}"
            )

        self.checkpointer.save(
            {"autonomous_module": self.model.autonomous_module.state_dict()},
            self.global_step, tag="stage3_final",
        )
        logger.info("Stage III IMT distillation complete")