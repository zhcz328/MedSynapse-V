"""
MedSAM3 Anatomical Encoder Wrapper
"""

from functools import partial
import hashlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision.transforms import functional as TF

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
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        feature_dim: int = 1024,
        mask_threshold: float = 0.7,
        target_spatial: Tuple[int, int] = (64, 64),
        backend: str = "auto",
        feature_backend: Optional[str] = None,
        sam3_model_name: str = "facebook/sam3",
        sam3_cache_dir: Optional[str] = None,
        medsam3_config_path: Optional[str] = None,
        medsam3_lora_path: Optional[str] = None,
        medsam3_repo_dir: Optional[str] = None,
        medsam3_script_path: Optional[str] = None,
        medsam3_prompt: str = "lesion",
        medsam3_threshold: float = 0.5,
        medsam3_nms_iou: float = 0.5,
        medsam3_resolution: int = 1024,
        medsam3_cache_dir: str = "cache/medsam3_masks",
        medsam3_python: str = "python3",
        medsam3_timeout_s: int = 300,
    ):
        super().__init__()
        self.mask_threshold = mask_threshold
        self.target_spatial = target_spatial
        self.feature_dim = feature_dim
        self._seg_head_loaded = False
        self.mask_backend = _normalize_backend(backend)
        self.feature_backend = _normalize_backend(feature_backend or backend)
        if self.mask_backend in {"medsam3", "medsam3_external"} and feature_backend is None:
            self.feature_backend = "sam3_hf"
        self.medsam3_masker = None

        self.encoder, self.encoder_backend = self._build_image_encoder(
            backend=self.feature_backend,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            out_chans=out_chans,
            feature_dim=feature_dim,
            sam3_model_name=sam3_model_name,
            sam3_cache_dir=sam3_cache_dir,
        )

        encoder_out_chans = feature_dim if self.encoder_backend == "sam3_hf" else out_chans
        self.seg_head = MedSAMSegmentationHead(in_channels=encoder_out_chans)
        if self.mask_backend in {"medsam3", "medsam3_external"}:
            self.medsam3_masker = MedSAM3ExternalMasker(
                config_path=medsam3_config_path,
                lora_path=medsam3_lora_path,
                repo_dir=medsam3_repo_dir,
                script_path=medsam3_script_path,
                default_prompt=medsam3_prompt,
                threshold=medsam3_threshold,
                nms_iou=medsam3_nms_iou,
                resolution=medsam3_resolution,
                cache_dir=medsam3_cache_dir,
                python_bin=medsam3_python,
                timeout_s=medsam3_timeout_s,
                target_spatial=target_spatial,
            )

        # Project neck output to target feature dimensionality
        if self.encoder_backend == "sam3_hf":
            self.feature_proj = nn.Identity()
        else:
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

        if pretrained_path is not None and self.encoder_backend != "sam3_hf":
            self._load_pretrained(pretrained_path)

        # Freeze all parameters
        self._freeze()

    def _build_image_encoder(
        self,
        backend: str,
        img_size: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        out_chans: int,
        feature_dim: int,
        sam3_model_name: str,
        sam3_cache_dir: Optional[str],
    ) -> Tuple[nn.Module, str]:
        backend = _normalize_backend(backend)
        if backend in {"sam3_hf", "sam3", "medsam3"}:
            return (
                Sam3HFVisionEncoder(
                    model_name=sam3_model_name,
                    feature_dim=feature_dim,
                    target_spatial=self.target_spatial,
                    cache_dir=sam3_cache_dir,
                ),
                "sam3_hf",
            )
        if backend in {"auto", "segment_anything", "sam", "medsam"}:
            try:
                from segment_anything.modeling import ImageEncoderViT

                return (
                    ImageEncoderViT(
                        depth=depth,
                        embed_dim=embed_dim,
                        img_size=img_size,
                        mlp_ratio=mlp_ratio,
                        norm_layer=partial(nn.LayerNorm, eps=1e-6),
                        num_heads=num_heads,
                        patch_size=patch_size,
                        qkv_bias=True,
                        use_rel_pos=True,
                        global_attn_indexes=[2, 5, 8, 11] if depth == 12 else (),
                        window_size=14,
                        out_chans=out_chans,
                    ),
                    "segment_anything",
                )
            except ImportError as exc:
                if backend != "auto":
                    raise RuntimeError(
                        "segment_anything is required for backend="
                        f"{backend!r}. Install MedSAM/SAM dependencies first."
                    ) from exc
                logger.warning(
                    "segment_anything is not installed; falling back to the local "
                    "ViT encoder. Real medsam_vit_b.pth checkpoints will not fully "
                    "load in fallback mode."
                )

        return (
            MedSAMImageEncoder(
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                out_chans=out_chans,
            ),
            "fallback",
        )

    def _load_pretrained(self, path: str):
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]

        loaded_encoder = False
        if any(k.startswith("image_encoder.") for k in state_dict):
            loaded_encoder = self._load_prefixed_module(
                self.encoder,
                state_dict,
                "image_encoder.",
                "MedSAM image_encoder",
            )
        elif any(k.startswith("encoder.") for k in state_dict):
            loaded_encoder = self._load_prefixed_module(
                self.encoder,
                state_dict,
                "encoder.",
                "MedSAM encoder",
            )

        if any(k.startswith("seg_head.") for k in state_dict):
            self._seg_head_loaded = self._load_prefixed_module(
                self.seg_head,
                state_dict,
                "seg_head.",
                "MedSAM segmentation head",
            )

        if not loaded_encoder:
            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            loaded_encoder = len(missing) < len(self.state_dict())
            if missing:
                logger.warning("Missing keys when loading MedSAM: %s", missing[:5])
            if unexpected:
                logger.warning("Unexpected keys when loading MedSAM: %s", unexpected[:5])

        if not loaded_encoder:
            logger.warning(
                "No compatible MedSAM encoder weights were loaded from %s. "
                "Check that the checkpoint matches the configured backend.",
                path,
            )
        if not self._seg_head_loaded:
            logger.info(
                "No compatible segmentation head found; causal masks will use "
                "deterministic feature-energy regions from the MedSAM encoder."
            )

    def _load_prefixed_module(
        self,
        module: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        name: str,
    ) -> bool:
        sub_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if not sub_state:
            return False
        missing, unexpected = module.load_state_dict(sub_state, strict=False)
        if missing:
            logger.warning("Missing keys when loading %s: %s", name, missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading %s: %s", name, unexpected[:5])
        loaded_keys = set(sub_state) - set(unexpected)
        return bool(loaded_keys)

    def _freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    @torch.no_grad()
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Resize for the configured encoder and normalize only when required."""
        if images.shape[-2:] != (self.encoder.img_size, self.encoder.img_size):
            images = F.interpolate(
                images,
                size=(self.encoder.img_size, self.encoder.img_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.encoder_backend != "sam3_hf":
            images = (images - self.pixel_mean) / self.pixel_std
        return images

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract flattened spatial features S ∈ R^{B × M × d_f}.
        M = H_f × W_f (target_spatial[0] * target_spatial[1]).
        """
        encoder_inputs = self.preprocess(images)
        neck_out = self.encoder(encoder_inputs)  # (B, out_chans, H_f, W_f)

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
    def extract_masks(
        self,
        images: torch.Tensor,
        concept_prompts: Optional[Tuple[str, ...]] = None,
    ) -> torch.Tensor:
        """
        Extract highest-confidence binary region masks B.
        B_bar (inverted) is used for causal counterfactual construction.

        Returns:
            masks: (B, 1, H_f, W_f) binary masks thresholded at tau
        """
        if self.medsam3_masker is not None:
            return self.medsam3_masker(images, concept_prompts=concept_prompts)

        encoder_inputs = self.preprocess(images)
        neck_out = self.encoder(encoder_inputs)
        return self._masks_from_features(neck_out)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        concept_prompts: Optional[Tuple[str, ...]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass returning both features and masks.

        Returns:
            dict with:
                features: (B, M, d_f) flattened spatial features
                masks:    (B, 1, H_f, W_f) binary region masks
                masks_flat: (B, M) flattened masks for element-wise masking
        """
        raw_images = images
        encoder_inputs = self.preprocess(raw_images)
        neck_out = self.encoder(encoder_inputs)

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
        if self.medsam3_masker is not None:
            masks = self.medsam3_masker(raw_images, concept_prompts=concept_prompts)
        else:
            masks = self._masks_from_features(neck_out, projected_features=features)
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

    def _masks_from_features(
        self,
        neck_out: torch.Tensor,
        projected_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._seg_head_loaded:
            logits = self.seg_head(neck_out)
            probs = torch.sigmoid(logits)
            if probs.shape[-2:] != tuple(self.target_spatial):
                probs = F.interpolate(
                    probs,
                    size=self.target_spatial,
                    mode="bilinear",
                    align_corners=False,
                )
            return (probs >= self.mask_threshold).float()

        features = projected_features
        if features is None:
            features = self.feature_proj(neck_out)
            if features.shape[-2:] != tuple(self.target_spatial):
                features = F.interpolate(
                    features,
                    size=self.target_spatial,
                    mode="bilinear",
                    align_corners=False,
                )
        energy = features.float().pow(2).mean(dim=1, keepdim=True).sqrt()
        flat = energy.flatten(1)
        min_v = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_v = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        normalized = (energy - min_v) / (max_v - min_v).clamp(min=1e-6)
        masks = (normalized >= self.mask_threshold).float()

        empty = masks.flatten(1).sum(dim=1) == 0
        if empty.any():
            flat_norm = normalized.flatten(1)
            top_idx = flat_norm.argmax(dim=1)
            masks_flat = masks.flatten(1)
            masks_flat[empty] = 0.0
            masks_flat[empty, top_idx[empty]] = 1.0
            masks = masks_flat.view_as(masks)

        return masks


class Sam3HFVisionEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/sam3",
        feature_dim: int = 1024,
        target_spatial: Tuple[int, int] = (64, 64),
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        try:
            from transformers import Sam3ImageProcessor, Sam3VisionModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers with SAM3 support is required for feature_backend=sam3_hf"
            ) from exc

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        self.model = Sam3VisionModel.from_pretrained(model_name, **kwargs)
        self.processor = Sam3ImageProcessor.from_pretrained(model_name, **kwargs)
        self.img_size = int(getattr(self.model.config.backbone_config, "image_size", 1008))
        self.patch_size = int(getattr(self.model.config.backbone_config, "patch_size", 14))
        hidden_size = int(getattr(self.model.config.backbone_config, "hidden_size", feature_dim))
        self.output_proj = nn.Identity() if hidden_size == feature_dim else nn.Linear(hidden_size, feature_dim)
        self.target_spatial = tuple(target_spatial)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pil_images = [self._to_pil(image) for image in images]
        encoded = self.processor(images=pil_images, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device=images.device, dtype=next(self.model.parameters()).dtype)
        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        hidden = self.output_proj(hidden)

        batch, seq_len, channels = hidden.shape
        grid_h = pixel_values.shape[-2] // self.patch_size
        grid_w = pixel_values.shape[-1] // self.patch_size
        if grid_h * grid_w != seq_len:
            grid_h = int(seq_len ** 0.5)
            grid_w = max(seq_len // max(grid_h, 1), 1)
        spatial = hidden[:, : grid_h * grid_w].view(batch, grid_h, grid_w, channels)
        spatial = spatial.permute(0, 3, 1, 2).contiguous()
        if spatial.shape[-2:] != self.target_spatial:
            spatial = F.interpolate(
                spatial,
                size=self.target_spatial,
                mode="bilinear",
                align_corners=False,
            )
        return spatial

    def _to_pil(self, image: torch.Tensor) -> Image.Image:
        image = image.detach().float().cpu()
        if image.max() > 2.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)
        return TF.to_pil_image(image).convert("RGB")


class MedSAM3ExternalMasker:
    """
    Bridge to the official MedSAM3/SAM3-LoRA inference script.

    The public MedSAM3-v1 release exposes text-guided masks through
    infer_sam.py. We keep feature extraction in-process and use this external
    bridge only for causal masks, with a file cache to avoid repeated CLI calls.
    """

    def __init__(
        self,
        config_path: Optional[str],
        lora_path: Optional[str],
        repo_dir: Optional[str],
        script_path: Optional[str],
        default_prompt: str,
        threshold: float,
        nms_iou: float,
        resolution: int,
        cache_dir: str,
        python_bin: str,
        timeout_s: int,
        target_spatial: Tuple[int, int],
    ):
        self.repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir else None
        if script_path:
            self.script_path = Path(script_path).expanduser().resolve()
        elif self.repo_dir:
            self.script_path = self.repo_dir / "infer_sam.py"
        else:
            self.script_path = None

        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.lora_path = Path(lora_path).expanduser().resolve() if lora_path else None
        self.default_prompt = default_prompt
        self.threshold = float(threshold)
        self.nms_iou = float(nms_iou)
        self.resolution = int(resolution)
        self.cache_dir = Path(cache_dir)
        self.python_bin = python_bin
        self.timeout_s = int(timeout_s)
        self.target_spatial = tuple(target_spatial)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __call__(
        self,
        images: torch.Tensor,
        concept_prompts: Optional[Tuple[str, ...]] = None,
    ) -> torch.Tensor:
        self._validate()
        masks = []
        for idx in range(images.shape[0]):
            prompt = self._prompt_for(idx, concept_prompts)
            masks.append(self._mask_one(images[idx], prompt).to(images.device))
        return torch.stack(masks, dim=0).to(dtype=images.dtype)

    def _validate(self):
        if self.script_path is None or not self.script_path.is_file():
            raise RuntimeError(
                "MedSAM3 backend is enabled but infer_sam.py was not found. "
                "Set encoder.medsam3.repo_dir or encoder.medsam3.script_path "
                "in configs/models/medsam_encoder_frozen.yaml after cloning "
                "https://github.com/Joey-S-Liu/MedSAM3."
            )
        if self.config_path is None or not self.config_path.is_file():
            raise RuntimeError(
                "MedSAM3 backend is enabled but configs/full_lora_config.yaml "
                "was not found. Set encoder.medsam3.config_path."
            )

    def _prompt_for(
        self,
        idx: int,
        concept_prompts: Optional[Tuple[str, ...]],
    ) -> str:
        if concept_prompts and idx < len(concept_prompts) and concept_prompts[idx]:
            return str(concept_prompts[idx])
        return self.default_prompt

    def _mask_one(self, image: torch.Tensor, prompt: str) -> torch.Tensor:
        key = self._cache_key(image, prompt)
        mask_path = self.cache_dir / f"{key}.png"
        if not mask_path.is_file():
            self._run_inference(image, prompt, mask_path)
        mask = Image.open(mask_path).convert("L").resize(
            self.target_spatial[::-1],
            Image.BILINEAR,
        )
        mask_t = TF.to_tensor(mask)
        return (mask_t >= self.threshold).float()

    def _run_inference(self, image: torch.Tensor, prompt: str, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="medsam3_") as tmpdir:
            tmpdir = Path(tmpdir)
            image_path = tmpdir / "image.png"
            config_path = self._runtime_config(tmpdir)
            self._save_image(image, image_path)
            cmd = [
                self.python_bin,
                str(self.script_path),
                "--config",
                str(config_path),
                "--image",
                str(image_path),
                "--prompt",
                prompt,
                "--threshold",
                str(self.threshold),
                "--nms-iou",
                str(self.nms_iou),
                "--output",
                str(output_path),
            ]
            subprocess.run(
                cmd,
                cwd=str(self.repo_dir or self.script_path.parent),
                check=True,
                timeout=self.timeout_s,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def _runtime_config(self, tmpdir: Path) -> Path:
        if self.lora_path is None:
            return self.config_path
        with open(self.config_path) as f:
            config = yaml.safe_load(f) or {}
        config.setdefault("output", {})["output_dir"] = str(self.lora_path)
        path = tmpdir / "medsam3_config.yaml"
        with open(path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return path

    def _save_image(self, image: torch.Tensor, image_path: Path):
        image = image.detach().float().cpu()
        if image.max() > 2.0:
            image = image / 255.0
        image = image.clamp(0.0, 1.0)
        TF.to_pil_image(image).convert("RGB").resize(
            (self.resolution, self.resolution),
            Image.BILINEAR,
        ).save(image_path)

    def _cache_key(self, image: torch.Tensor, prompt: str) -> str:
        image_cpu = image.detach().cpu()
        payload = image_cpu.numpy().tobytes()
        digest = hashlib.sha1()
        digest.update(payload)
        digest.update(prompt.encode("utf-8"))
        digest.update(str(self.threshold).encode("utf-8"))
        digest.update(str(self.nms_iou).encode("utf-8"))
        return digest.hexdigest()


def _normalize_backend(backend: Optional[str]) -> str:
    return (backend or "auto").lower().replace("-", "_")
