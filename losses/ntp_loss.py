import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NTPLoss(nn.Module):
    """
    Cross-entropy loss for next-token prediction with label masking.
    """

    def __init__(
        self,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction=reduction,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, L, V) model output logits over vocabulary
            labels: (B, L) target token IDs with -100 for masked positions
            memory_mask: optional (B, N) mask indicating active memory tokens

        Returns:
            scalar loss
        """
        # Shift for autoregressive prediction: predict next token
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross-entropy
        B, L, V = shift_logits.shape
        loss = self.loss_fn(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        )

        return loss

    @torch.no_grad()
    def compute_token_accuracy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Compute per-token accuracy over non-masked positions."""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != self.ignore_index
        correct = ((preds == shift_labels) & mask).sum()
        total = mask.sum()

        return (correct / total).item() if total > 0 else 0.0
