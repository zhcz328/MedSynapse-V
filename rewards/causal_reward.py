import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List


class CausalCounterfactualReward:
    def __init__(
        self,
        normalize: bool = True,
        clip_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
    ):
        self.normalize = normalize
        self.clip_range = clip_range

    @torch.no_grad()
    def compute_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory: torch.Tensor,
        labels: torch.Tensor,
        injector: nn.Module,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities conditioned on given memory.

        Args:
            model:          VLM backbone
            input_ids:      (B, L) token IDs
            attention_mask: (B, L) attention mask
            memory:         (B, N, d_h) diagnostic memory
            labels:         (B, L) generation labels for log-prob computation
            injector:       HiddenStreamInjector instance
            inputs_embeds:  precomputed input embeddings (optional)

        Returns:
            log_probs: (B, T) per-token log probabilities over generated tokens
        """
        if inputs_embeds is None:
            inputs_embeds = model.get_input_embeddings()(input_ids)

        # Inject memory into embedding sequence
        injected = injector.inject(
            inputs_embeds=inputs_embeds,
            memory=memory,
            attention_mask=attention_mask,
        )

        outputs = model(
            inputs_embeds=injected["inputs_embeds"],
            attention_mask=injected["attention_mask"],
            return_dict=True,
        )

        logits = outputs.logits

        # Extract log probabilities for actual generated tokens
        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding
        token_mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * token_mask

        return token_log_probs, token_mask

    def compute(
        self,
        log_probs_original: torch.Tensor,
        log_probs_intervened: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute r_causal as the sum of log-likelihood ratios.

        r_causal(o) = sum_t [log pi(o_t | M) - log pi(o_t | M')]

        Args:
            log_probs_original:   (B, T) log probs with original memory M
            log_probs_intervened: (B, T) log probs with intervened memory M'
            token_mask:           (B, T) mask for valid generated tokens

        Returns:
            rewards: (B,) causal counterfactual rewards
        """
        # Per-token log ratio
        log_ratio = log_probs_original - log_probs_intervened  # (B, T)
        log_ratio = log_ratio * token_mask

        # Sum over sequence
        rewards = log_ratio.sum(dim=-1)  # (B,)

        if self.normalize:
            seq_lengths = token_mask.sum(dim=-1).clamp(min=1)
            rewards = rewards / seq_lengths

        if self.clip_range is not None:
            rewards = rewards.clamp(min=self.clip_range[0], max=self.clip_range[1])

        return rewards

    def compute_composite_reward(
        self,
        r_acc: torch.Tensor,
        r_causal: torch.Tensor,
        lambda_acc: float = 1.0,
        lambda_causal: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute composite reward R(o) = lambda_acc * r_acc + lambda_causal * r_causal.

        Args:
            r_acc:           (B,) or (G,) accuracy rewards
            r_causal:        (B,) or (G,) causal counterfactual rewards
            lambda_acc:      weight for accuracy component
            lambda_causal:   weight for causal component

        Returns:
            composite: (B,) or (G,) composite rewards
        """
        return lambda_acc * r_acc + lambda_causal * r_causal
