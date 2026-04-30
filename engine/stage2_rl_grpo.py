"""
Stage II: Causal Counterfactual Refinement — GRPO Engine (§2.3).

Performs policy optimization within the M-conditioned latent space using
Group Relative Policy Optimization (GRPO). Freezes P_phi and optimizes
LoRA adapters on the VLM backbone with composite rewards:

  R(o) = lambda_acc * r_acc + lambda_causal * r_causal

where r_acc is the diagnostic accuracy reward (Eq. 4) and r_causal is
the causal counterfactual reward (Eq. 5) computed via region-level
feature masking interventions from MedSAM3.

Policy objective J_CCR(theta) follows Eq. 3 with clipped surrogate,
group-normalized advantages, and KL divergence penalty.

Hyperparameters (Appendix Table 1):
  LR=1e-5, LoRA r=64/alpha=128, G=4 trajectories, eps=0.2,
  lambda_acc=1.0, lambda_causal=0.5, BS=32 rollout, 200 steps
"""

import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from copy import deepcopy

from losses.grpo_loss import GRPOLoss
from rewards.accuracy_reward import AccuracyReward
from rewards.causal_reward import CausalCounterfactualReward
from engine.lr_scheduler import build_optimizer_and_scheduler
from utils.checkpointer import CheckpointManager

logger = logging.getLogger(__name__)


class Stage2GRPOEngine:
    """
    GRPO training engine for Stage II: Causal Counterfactual Refinement.

    For each training sample, generates G candidate trajectories via
    sampling, computes composite rewards, estimates group-normalized
    advantages, and updates LoRA adapters via the clipped surrogate
    objective conditioned on diagnostic implicit memory M.
    """

    def __init__(
        self,
        model,
        processor,
        train_dataloader: DataLoader,
        config: Optional[Dict] = None,
        output_dir: str = "outputs/stage2",
        device: torch.device = None,
        local_rank: int = -1,
    ):
        self.model = model
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.config = config or {}
        self.output_dir = output_dir
        self.device = device or torch.device("cuda")
        self.local_rank = local_rank

        os.makedirs(output_dir, exist_ok=True)

        grpo_cfg = self.config.get("grpo", {})
        self.group_size = grpo_cfg.get("group_size", 4)
        self.clip_eps = grpo_cfg.get("clip_eps", 0.2)
        self.kl_coeff = grpo_cfg.get("kl_coeff", 0.02)
        self.temperature = grpo_cfg.get("temperature", 0.7)
        self.top_p = grpo_cfg.get("top_p", 0.95)
        self.max_gen_len = grpo_cfg.get("max_generation_length", 1024)
        self.advantage_eps = grpo_cfg.get("advantage_eps", 1e-6)

        reward_cfg = self.config.get("reward", {})
        self.lambda_acc = reward_cfg.get("lambda_acc", 1.0)
        self.lambda_causal = reward_cfg.get("lambda_causal", 0.5)

        train_cfg = self.config.get("training", {})
        self.total_steps = train_cfg.get("training_steps", 200)
        self.gradient_accumulation_steps = train_cfg.get("gradient_accumulation_steps", 2)
        self.max_grad_norm = train_cfg.get("max_grad_norm", 1.0)
        self.use_amp = train_cfg.get("mixed_precision", "bf16") != "fp32"
        self.amp_dtype = torch.bfloat16

        log_cfg = self.config.get("logging", {})
        self.log_interval = log_cfg.get("log_interval", 10)
        self.save_interval = log_cfg.get("save_interval", 50)

        self.grpo_loss = GRPOLoss(
            clip_eps=self.clip_eps,
            kl_coeff=self.kl_coeff,
            advantage_eps=self.advantage_eps,
        )
        self.accuracy_reward = AccuracyReward()
        self.causal_reward = CausalCounterfactualReward()

        self._setup_optimizer()
        self.checkpointer = CheckpointManager(output_dir)

        # Store reference policy for KL computation
        self.ref_model = None

    def _setup_optimizer(self):
        train_cfg = self.config.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        sched_cfg = train_cfg.get("scheduler", {})

        trainable_params = [
            (n, p) for n, p in self.model.vlm.named_parameters() if p.requires_grad
        ]
        total = sum(p.numel() for _, p in trainable_params)
        logger.info(f"Stage II trainable parameters (LoRA): {total / 1e6:.2f}M")

        warmup_steps = int(self.total_steps * sched_cfg.get("warmup_ratio", 0.05))

        self.optimizer, self.scheduler = build_optimizer_and_scheduler(
            params=trainable_params,
            lr=opt_cfg.get("lr", 1e-5),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def _snapshot_reference_policy(self):
        """Snapshot current policy weights for importance ratio computation."""
        self.ref_model = {}
        for n, p in self.model.vlm.named_parameters():
            if p.requires_grad:
                self.ref_model[n] = p.data.clone()

    def _restore_reference_policy(self):
        """Swap in reference weights temporarily for log-prob computation."""
        current_weights = {}
        for n, p in self.model.vlm.named_parameters():
            if p.requires_grad and n in self.ref_model:
                current_weights[n] = p.data.clone()
                p.data.copy_(self.ref_model[n])
        return current_weights

    def _restore_current_policy(self, current_weights: Dict):
        for n, p in self.model.vlm.named_parameters():
            if n in current_weights:
                p.data.copy_(current_weights[n])

    @torch.no_grad()
    def _generate_trajectories(
        self,
        batch: Dict,
        memory: torch.Tensor,
    ) -> List[Dict]:
        """
        Sample G candidate trajectories per sample using current policy.

        Returns list of dicts with keys: input_ids, generated_ids,
        generated_text, attention_mask
        """
        self.model.vlm.eval()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        B = input_ids.shape[0]

        # Get embeddings and inject memory
        inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)
        injected = self.model.injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory,
            attention_mask=attention_mask,
        )

        trajectories = []
        for g in range(self.group_size):
            outputs = self.model.vlm.generate(
                inputs_embeds=injected["inputs_embeds"],
                attention_mask=injected["attention_mask"],
                max_new_tokens=self.max_gen_len,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False,
            )

            gen_ids = outputs.sequences[:, injected["inputs_embeds"].shape[1]:]
            gen_texts = self.processor.tokenizer.batch_decode(
                gen_ids, skip_special_tokens=True
            )

            trajectories.append({
                "generated_ids": gen_ids,
                "generated_text": gen_texts,
                "full_sequences": outputs.sequences,
            })

        self.model.vlm.train()
        return trajectories

    def _compute_log_probs_for_trajectory(
        self,
        batch: Dict,
        memory: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probabilities for a generated trajectory."""
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        inputs_embeds = self.model.vlm.get_input_embeddings()(input_ids)
        injected = self.model.injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory,
            attention_mask=attention_mask,
        )

        # Concatenate injected context with generated tokens
        gen_embeds = self.model.vlm.get_input_embeddings()(generated_ids)
        full_embeds = torch.cat([injected["inputs_embeds"], gen_embeds], dim=1)
        full_mask = torch.cat([
            injected["attention_mask"],
            torch.ones_like(generated_ids, dtype=injected["attention_mask"].dtype),
        ], dim=1)

        outputs = self.model.vlm(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            return_dict=True,
        )

        # Extract log probs for generated portion
        context_len = injected["inputs_embeds"].shape[1]
        gen_logits = outputs.logits[:, context_len - 1:-1, :]
        log_probs = F.log_softmax(gen_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=generated_ids.unsqueeze(-1)
        ).squeeze(-1)

        gen_mask = (generated_ids != self.processor.tokenizer.pad_token_id).float()
        return token_log_probs * gen_mask, gen_mask

    @torch.no_grad()
    def _compute_causal_reward(
        self,
        batch: Dict,
        memory: torch.Tensor,
        memory_intervened: torch.Tensor,
        generated_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_causal (Eq. 5) as the log-likelihood ratio between
        original memory M and intervened memory M'.
        """
        log_probs_orig, mask = self._compute_log_probs_for_trajectory(
            batch, memory, generated_ids
        )
        log_probs_inter, _ = self._compute_log_probs_for_trajectory(
            batch, memory_intervened, generated_ids
        )

        return self.causal_reward.compute(log_probs_orig, log_probs_inter, mask)

    def _train_step(self, batch: Dict) -> Dict[str, float]:
        """Single GRPO training step with composite rewards."""
        images = batch.get("pixel_values")
        if images is not None:
            images = images.to(self.device)

        ground_truths = batch.get("answer", batch.get("labels_text", []))
        task_types = batch.get("task_type", ["closed_ended"] * len(ground_truths))

        # Generate privileged memory and counterfactual memory
        with torch.no_grad():
            mem_output = self.model.generate_memory_privileged(images)
            memory = mem_output["memory"]
            memory_intervened = self.model.generate_memory_counterfactual(images)

        # Snapshot reference policy
        self._snapshot_reference_policy()

        # Generate G trajectories
        trajectories = self._generate_trajectories(batch, memory)

        all_policy_losses = []
        all_rewards = []

        # Compute rewards for all trajectories
        trajectory_rewards = []
        for traj in trajectories:
            # r_acc
            r_acc = torch.tensor(
                self.accuracy_reward(traj["generated_text"], ground_truths, task_types),
                device=self.device, dtype=torch.float32,
            )

            # r_causal
            r_causal = self._compute_causal_reward(
                batch, memory, memory_intervened, traj["generated_ids"]
            )

            composite = self.lambda_acc * r_acc + self.lambda_causal * r_causal
            trajectory_rewards.append(composite)
            all_rewards.append(composite.mean().item())

        # Stack rewards: (G, B) -> compute per-sample advantages
        rewards_tensor = torch.stack(trajectory_rewards, dim=0)  # (G, B)
        advantages = self.grpo_loss.compute_advantages(rewards_tensor.mean(dim=1))

        # Policy gradient update
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for g, traj in enumerate(trajectories):
            # Current policy log probs
            curr_log_probs, seq_mask = self._compute_log_probs_for_trajectory(
                batch, memory, traj["generated_ids"]
            )

            # Reference policy log probs
            saved = self._restore_reference_policy()
            with torch.no_grad():
                old_log_probs, _ = self._compute_log_probs_for_trajectory(
                    batch, memory, traj["generated_ids"]
                )
            self._restore_current_policy(saved)

            loss_dict = self.grpo_loss(
                log_probs=curr_log_probs,
                old_log_probs=old_log_probs,
                advantages=advantages[g:g+1].expand(curr_log_probs.shape[0]),
                sequence_mask=seq_mask,
            )
            total_loss = total_loss + loss_dict["loss"] / self.group_size

        return {
            "loss": total_loss,
            "mean_reward": sum(all_rewards) / len(all_rewards),
            "mean_r_acc": trajectory_rewards[0].mean().item() if trajectory_rewards else 0,
        }

    def train(self):
        logger.info(f"Starting Stage II CCR-GRPO: {self.total_steps} steps")
        self.model.vlm.train()
        self.model.memory_sampler.eval()

        data_iter = iter(self.train_dataloader)
        running_reward = 0.0

        for step in range(1, self.total_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            self.optimizer.zero_grad()

            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                result = self._train_step(batch)
                loss = result["loss"] / self.gradient_accumulation_steps

            loss.backward()

            if step % self.gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.vlm.parameters() if p.requires_grad],
                    self.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            running_reward = 0.95 * running_reward + 0.05 * result["mean_reward"]

            if step % self.log_interval == 0:
                logger.info(
                    f"Step {step}/{self.total_steps} | "
                    f"Loss: {result['loss'].item():.4f} | "
                    f"Reward: {running_reward:.4f} | "
                    f"R_acc: {result['mean_r_acc']:.3f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                )

            if step % self.save_interval == 0:
                self.checkpointer.save(
                    {
                        "vlm_lora": {
                            n: p.data for n, p in self.model.vlm.named_parameters()
                            if p.requires_grad
                        },
                    },
                    step, tag="stage2",
                )

        # Final save
        self.checkpointer.save(
            {
                "vlm_lora": {
                    n: p.data for n, p in self.model.vlm.named_parameters()
                    if p.requires_grad
                },
            },
            self.total_steps, tag="stage2_final",
        )
        logger.info("Stage II CCR-GRPO complete")