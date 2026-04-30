"""
MedSAM3 Anatomical Encoder Wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MedSAMImageEncoder(nn.Module):
    """
    ViT-B image encoder adapted from the Segment Anything architecture.
    Extracts multi-scale spatial features from medical images.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding via convolution
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size,
        )

        # Positional embedding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Neck: project to output channels with spatial reshape
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            nn.LayerNorm([out_chans, img_size // patch_size, img_size // patch_size]),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm([out_chans, img_size // patch_size, img_size // patch_size]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        # Reshape back to spatial
        x = x.transpose(1, 2).view(B, self.embed_dim, H, W)
        x = self.neck(x)
        return x


class ViTBlock(nn.Module):
    """Standard ViT Transformer block with pre-norm."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class MedSAMSegmentationHead(nn.Module):
    """Lightweight mask decoder for region proposals."""

    def __init__(self, in_channels: int = 256, hidden_dim: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.conv3(x)


class MedSAMWrapper(nn.Module):
    """
    Complete MedSAM3 wrapper providing:
      - Spatial feature extraction F ∈ R^{B × (H_f*W_f) × d_f}
      - Binary region masks B for causal counterfactual reward

    All parameters are frozen; gradients do not flow through this module.
    """

    def __init__(
        self,
        pretrained_path: Optional[str] = None,
        img_size: int = 1024,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        out_chans: int = 256,
        feature_dim: int = 1024,
        mask_threshold: float = 0.7,
        target_spatial: Tuple[int, int] = (64, 64),
    ):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.target_spatial = target_spatial
        self.feature_dim = feature_dim

        self.encoder = MedSAMImageEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            out_chans=out_chans,
        )

        self.seg_head = MedSAMSegmentationHead(in_channels=out_chans)

        # Project neck output to target feature dimensionality
        self.feature_proj = nn.Sequential(
            nn.Conv2d(out_chans, feature_dim, kernel_size=1),
            nn.GroupNorm(32, feature_dim),
            nn.GELU(),
        )

        # Preprocessing constants
        self.register_buffer(
            "pixel_mean",
            torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1),
        )

        if pretrained_path is not None:
            self._load_pretrained(pretrained_path)

        # Freeze all parameters
        self._freeze()

    def _load_pretrained(self, path: str):
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys when loading MedSAM3: {missing[:5]}...")
        if unexpected:
            logger.warning(f"Unexpected keys when loading MedSAM3: {unexpected[:5]}...")

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize to target size and normalize."""
        if images.shape[-2:] != (self.encoder.img_size, self.encoder.img_size):
            images = F.interpolate(
                images,
                size=(self.encoder.img_size, self.encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.pixel_mean) / self.pixel_std
        return images

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract flattened spatial features S ∈ R^{B × M × d_f}.
        M = H_f × W_f (target_spatial[0] * target_spatial[1]).
        """
        images = self.preprocess(images)
        neck_out = self.encoder(images)  # (B, out_chans, H_f, W_f)

        features = self.feature_proj(neck_out)  # (B, d_f, H_f, W_f)

        # Interpolate to target spatial resolution if needed
        if features.shape[-2:] != tuple(self.target_spatial):
            features = F.interpolate(
                features,
                size=self.target_spatial,
                mode="bilinear",
                align_corners=False,
            )

        B, C, H, W = features.shape
        features = features.flatten(2).transpose(1, 2)  # (B, M, d_f)
        return features

    @torch.no_grad()
    def extract_masks(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract highest-confidence binary region masks B.
        B_bar (inverted) is used for causal counterfactual construction.

        Returns:
            masks: (B, 1, H_f, W_f) binary masks thresholded at tau
        """
        images = self.preprocess(images)
        neck_out = self.encoder(images)
        logits = self.seg_head(neck_out)  # (B, 1, H_f, W_f)
        probs = torch.sigmoid(logits)

        # Interpolate to target spatial
        if probs.shape[-2:] != tuple(self.target_spatial):
            probs = F.interpolate(
                probs,
                size=self.target_spatial,
                mode="bilinear",
                align_corners=False,
            )

        masks = (probs >= self.mask_threshold).float()
        return masks

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass returning both features and masks.

        Returns:
            dict with:
                features: (B, M, d_f) flattened spatial features
                masks:    (B, 1, H_f, W_f) binary region masks
                masks_flat: (B, M) flattened masks for element-wise masking
        """
        images = self.preprocess(images)
        neck_out = self.encoder(images)

        # Features
        features = self.feature_proj(neck_out)
        if features.shape[-2:] != tuple(self.target_spatial):
            features = F.interpolate(
                features, size=self.target_spatial,
                mode="bilinear", align_corners=False,
            )
        B, C, H, W = features.shape
        features_flat = features.flatten(2).transpose(1, 2)  # (B, M, d_f)

        # Masks
        logits = self.seg_head(neck_out)
        probs = torch.sigmoid(logits)
        if probs.shape[-2:] != tuple(self.target_spatial):
            probs = F.interpolate(
                probs, size=self.target_spatial,
                mode="bilinear", align_corners=False,
            )
        masks = (probs >= self.mask_threshold).float()  # (B, 1, H, W)
        masks_flat = masks.flatten(2).squeeze(1)  # (B, M)

        return {
            "features": features_flat,
            "masks": masks,
            "masks_flat": masks_flat,
        }

    def construct_counterfactual(
        self,
        features: torch.Tensor,
        masks_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct intervened features by zeroing out masked regions.
        F' = F ⊙ B_bar, where B_bar is the inverted mask.

        Args:
            features:   (B, M, d_f) original features
            masks_flat: (B, M) binary masks (1 = diagnostic region)

        Returns:
            features_intervened: (B, M, d_f) with diagnostic regions zeroed
        """
        # Invert: B_bar = 1 - B
        inverted_masks = 1.0 - masks_flat  # (B, M)
        inverted_masks = inverted_masks.unsqueeze(-1)  # (B, M, 1)
        return features * inverted_masks
