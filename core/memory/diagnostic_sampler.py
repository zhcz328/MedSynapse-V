import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttentionBlock(nn.Module):
    """Single cross-attention block with pre-norm residual connections."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # Self-attention among probes
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.self_attn_drop = nn.Dropout(dropout)

        # Cross-attention: probes attend to encoder features
        self.cross_attn_norm_q = nn.LayerNorm(d_model)
        self.cross_attn_norm_kv = nn.LayerNorm(d_model)
        self.cross_attn_q = nn.Linear(d_model, d_model)
        self.cross_attn_k = nn.Linear(d_model, d_model)
        self.cross_attn_v = nn.Linear(d_model, d_model)
        self.cross_attn_out = nn.Linear(d_model, d_model)
        self.cross_attn_drop = nn.Dropout(dropout)

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def _multihead_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_out: nn.Linear,
        drop: nn.Dropout,
    ) -> torch.Tensor:
        B, Lq, _ = q.shape
        Lk = k.shape[1]

        q = q.view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        scale = math.sqrt(self.d_k)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = drop(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return proj_out(out)

    def forward(
        self,
        probes: torch.Tensor,
        encoder_seq: torch.Tensor,
    ) -> torch.Tensor:
        # Self-attention among probes
        h = self.self_attn_norm(probes)
        q = self.self_attn_q(h)
        k = self.self_attn_k(h)
        v = self.self_attn_v(h)
        probes = probes + self._multihead_attn(q, k, v, self.self_attn_out, self.self_attn_drop)

        # Cross-attention: probes query encoder features
        h_q = self.cross_attn_norm_q(probes)
        h_kv = self.cross_attn_norm_kv(encoder_seq)
        q = self.cross_attn_q(h_q)
        k = self.cross_attn_k(h_kv)
        v = self.cross_attn_v(h_kv)
        probes = probes + self._multihead_attn(q, k, v, self.cross_attn_out, self.cross_attn_drop)

        # FFN
        probes = probes + self.ffn(self.ffn_norm(probes))
        return probes


class DiagnosticMemorySampler(nn.Module):
    """
    P_phi: Condenses high-dimensional anatomical encoder features into
    N compact diagnostic implicit memory vectors via learnable meta-query
    probes and L-layer cross-attention.

    Args:
        num_probes:   N, number of diagnostic implicit tokens
        d_encoder:    d_f, encoder feature dimensionality
        d_hidden:     d_h, VLM hidden state dimensionality
        num_layers:   L, number of cross-attention Transformer layers
        num_heads:    attention heads per layer
        ffn_dim:      feed-forward intermediate dimensionality
        probe_init_std: std for truncated normal initialization of Q_0
    """

    def __init__(
        self,
        num_probes: int = 16,
        d_encoder: int = 1024,
        d_hidden: int = 4096,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 4096,
        probe_init_std: float = 0.02,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_probes = num_probes
        self.d_encoder = d_encoder
        self.d_hidden = d_hidden

        # Learnable meta-query probes Q_0 ∈ R^{N × d_f}
        self.probes = nn.Parameter(
            torch.empty(num_probes, d_encoder)
        )
        nn.init.trunc_normal_(self.probes, std=probe_init_std)

        # L-layer cross-attention Transformer
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                d_model=d_encoder,
                n_heads=num_heads,
                d_ff=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_encoder)

        # Linear projection d_f -> d_h
        self.output_proj = nn.Linear(d_encoder, d_hidden)

    def forward(
        self,
        encoder_features: torch.Tensor,
        return_pre_proj: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            encoder_features: S ∈ R^{B × M × d_f}, flattened spatial features
            return_pre_proj: if True, also return features before output projection

        Returns:
            M ∈ R^{B × N × d_h}, diagnostic implicit memory
        """
        B = encoder_features.shape[0]

        # Broadcast probes across batch
        probes = self.probes.unsqueeze(0).expand(B, -1, -1)

        # Pass through L cross-attention layers
        for layer in self.layers:
            probes = layer(probes, encoder_features)

        probes = self.final_norm(probes)

        if return_pre_proj:
            pre_proj = probes
            memory = self.output_proj(probes)
            return memory, pre_proj

        memory = self.output_proj(probes)
        return memory

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
