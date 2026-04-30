import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class JSDLoss(nn.Module):
    def __init__(
        self,
        beta: float = 0.5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.beta = beta
        self.temperature = temperature

    def jsd_beta(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute generalized JSD_beta between teacher and student distributions.

        Args:
            teacher_logits: (B, V) teacher next-token logits
            student_logits: (B, V) student next-token logits

        Returns:
            jsd: (B,) per-sample JSD values
        """
        # Compute distributions with temperature scaling
        p = F.softmax(teacher_logits / self.temperature, dim=-1)
        q = F.softmax(student_logits / self.temperature, dim=-1)

        # Mixture distribution m_bar = beta * pi^+ + (1 - beta) * pi^-
        m = self.beta * p + (1.0 - self.beta) * q

        # Clamp for numerical stability
        m = m.clamp(min=1e-8)
        p = p.clamp(min=1e-8)
        q = q.clamp(min=1e-8)

        # JSD_beta = beta * KL(p || m) + (1 - beta) * KL(q || m)
        kl_p_m = F.kl_div(m.log(), p, reduction="none").sum(dim=-1)
        kl_q_m = F.kl_div(m.log(), q, reduction="none").sum(dim=-1)

        jsd = self.beta * kl_p_m + (1.0 - self.beta) * kl_q_m
        return jsd

    def forward(
        self,
        teacher_logits: torch.Tensor,
        student_logits: torch.Tensor,
        sequence_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute trajectory-averaged per-position JSD loss.

        Args:
            teacher_logits: (B, L, V) pi^+(· | X, q, M_pri, y_hat_{<n})
            student_logits: (B, L, V) pi^-(· | X, q, M_auto, y_hat_{<n})
            sequence_mask:  (B, L) mask for valid generation positions

        Returns:
            dict with: loss, mean_jsd, output_agreement
        """
        B, L, V = teacher_logits.shape

        # Compute per-position JSD
        teacher_flat = teacher_logits.reshape(B * L, V)
        student_flat = student_logits.reshape(B * L, V)
        jsd_flat = self.jsd_beta(teacher_flat, student_flat)
        jsd_per_pos = jsd_flat.view(B, L)  # (B, L)

        # Apply sequence mask and average
        if sequence_mask is not None:
            jsd_per_pos = jsd_per_pos * sequence_mask
            seq_lengths = sequence_mask.sum(dim=-1).clamp(min=1)
            per_sample_jsd = jsd_per_pos.sum(dim=-1) / seq_lengths
        else:
            per_sample_jsd = jsd_per_pos.mean(dim=-1)

        loss = per_sample_jsd.mean()

        # Temperature compensation
        if self.temperature != 1.0:
            loss = loss * (self.temperature ** 2)

        # Diagnostics
        with torch.no_grad():
            teacher_preds = teacher_logits.argmax(dim=-1)
            student_preds = student_logits.argmax(dim=-1)
            if sequence_mask is not None:
                agreement = ((teacher_preds == student_preds) & sequence_mask.bool()).sum()
                total = sequence_mask.sum()
            else:
                agreement = (teacher_preds == student_preds).sum()
                total = torch.tensor(B * L, dtype=torch.float, device=loss.device)
            output_agreement = agreement.float() / total.clamp(min=1)

        return {
            "loss": loss,
            "mean_jsd": per_sample_jsd.mean().detach(),
            "output_agreement": output_agreement,
        }
