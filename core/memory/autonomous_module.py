import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AutonomousMemoryModule(nn.Module):

    def __init__(
        self,
        d_input: int = 4096,
        d_hidden: int = 4096,
        d_output: int = 4096,
        num_tokens: int = 16,
        num_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_output = d_output

        # Pooling projection from variable-length visual features
        self.pool_proj = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.GELU() if activation == "gelu" else nn.SiLU(),
            nn.LayerNorm(d_hidden),
        )

        # MLP layers to generate N × d_h memory
        layers = []
        in_dim = d_hidden
        for i in range(num_layers):
            out_dim = d_hidden if i < num_layers - 1 else num_tokens * d_output
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
            ])
            if i < num_layers - 1:
                layers.append(nn.GELU() if activation == "gelu" else nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Per-token normalization for output stability
        self.output_norm = nn.LayerNorm(d_output)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        visual_features: torch.Tensor,
        question_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            visual_features:   VLM visual encoder output, (B, L_v, d_input)
            question_features: optional VLM question encoding, (B, L_q, d_input)

        Returns:
            M_auto ∈ R^{B × N × d_h}
        """
        # Adaptive average pooling over sequence length
        pooled = visual_features.mean(dim=1)  # (B, d_input)

        # Optionally condition on question encoding
        if question_features is not None:
            q_pooled = question_features.mean(dim=1)
            pooled = pooled + q_pooled

        h = self.pool_proj(pooled)     # (B, d_hidden)
        out = self.mlp(h)              # (B, N * d_output)

        # Reshape to (B, N, d_h) and normalize
        memory = out.view(-1, self.num_tokens, self.d_output)
        memory = self.output_norm(memory)

        return memory

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
