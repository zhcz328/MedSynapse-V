import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization loss.

    Computes the clipped policy gradient objective conditioned on
    diagnostic implicit memory M, with group-level advantage normalization.

    Args:
        clip_eps:       PPO-style clipping coefficient epsilon
        kl_coeff:       KL divergence penalty coefficient beta
        advantage_eps:  numerical stability constant for advantage normalization
    """

    def __init__(
        self,
        clip_eps: float = 0.2,
        kl_coeff: float = 0.02,
        advantage_eps: float = 1e-6,
    ):
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.advantage_eps = advantage_eps

    def compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute group-normalized advantages.
        A_hat_i = (R(o_i) - mu_G) / (sigma_G + eps)

        Args:
            rewards: (G,) reward for each trajectory in the group

        Returns:
            advantages: (G,) normalized advantages
        """
        mu = rewards.mean()
        sigma = rewards.std()
        return (rewards - mu) / (sigma + self.advantage_eps)

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            log_probs:     (G, T) log pi_theta(o_t | ..., M, o_{<t})
            old_log_probs: (G, T) log pi_theta_old(o_t | ..., M, o_{<t})
            advantages:    (G,) group-normalized advantages
            sequence_mask: (G, T) mask for valid tokens
            ref_log_probs: (G, T) log pi_ref for KL penalty (optional)

        Returns:
            dict with: loss, policy_loss, kl_loss, clip_fraction
        """
        G, T = log_probs.shape

        # Importance sampling ratio rho_{i,t}
        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)

        # Expand advantages to per-token
        adv_expanded = advantages.unsqueeze(-1).expand_as(ratio)  # (G, T)

        # Clipped surrogate objective
        surr1 = ratio * adv_expanded
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_expanded

        # Per-token loss: min of clipped and unclipped
        token_loss = torch.min(surr1, surr2)

        # Apply sequence mask
        if sequence_mask is not None:
            token_loss = token_loss * sequence_mask
            num_tokens = sequence_mask.sum()
        else:
            num_tokens = torch.tensor(G * T, dtype=torch.float, device=log_probs.device)

        # Mean over tokens, then over group
        per_trajectory_loss = token_loss.sum(dim=-1)
        if sequence_mask is not None:
            per_trajectory_lengths = sequence_mask.sum(dim=-1).clamp(min=1)
            per_trajectory_loss = per_trajectory_loss / per_trajectory_lengths

        policy_loss = -per_trajectory_loss.mean()

        # KL divergence penalty
        kl_loss = torch.tensor(0.0, device=log_probs.device)
        if ref_log_probs is not None and self.kl_coeff > 0:
            kl_div = log_probs - ref_log_probs  # approximate KL
            if sequence_mask is not None:
                kl_div = kl_div * sequence_mask
            kl_loss = kl_div.sum() / num_tokens.clamp(min=1)

        total_loss = policy_loss + self.kl_coeff * kl_loss

        # Diagnostics
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float()
            if sequence_mask is not None:
                clip_fraction = (clip_fraction * sequence_mask).sum() / num_tokens.clamp(min=1)
            else:
                clip_fraction = clip_fraction.mean()

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "clip_fraction": clip_fraction,
            "mean_ratio": ratio.mean().detach(),
            "mean_advantage": advantages.mean().detach(),
        }
