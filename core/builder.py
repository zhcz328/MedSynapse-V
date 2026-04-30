"""
MedSynapse-V Model Builder
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging
import yaml

from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
)

from core.memory import (
    DiagnosticMemorySampler,
    AutonomousMemoryModule,
    HiddenStreamInjector,
)
from core.encoders.medsam_wrapper import MedSAMWrapper
from core.encoders.qwen_vision import QwenVisionFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class MedSynapseVConfig:
    """Unified configuration for MedSynapse-V."""
    # VLM
    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_dtype: str = "bfloat16"
    vlm_attn_impl: str = "flash_attention_2"
    # Encoder
    encoder_path: str = "checkpoints/medsam3_vit_b.pth"
    encoder_img_size: int = 1024
    encoder_feature_dim: int = 1024
    encoder_spatial: tuple = (64, 64)
    mask_threshold: float = 0.7
    # Memory sampler
    num_probes: int = 16
    sampler_layers: int = 2
    sampler_heads: int = 8
    sampler_ffn_dim: int = 4096
    probe_init_std: float = 0.02
    # Autonomous module
    auto_hidden_dim: int = 4096
    auto_num_layers: int = 2
    # LoRA
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    @classmethod
    def from_yaml(cls, model_config_path: str, stage_config_path: str) -> "MedSynapseVConfig":
        config = cls()
        with open(model_config_path) as f:
            model_cfg = yaml.safe_load(f)
        with open(stage_config_path) as f:
            stage_cfg = yaml.safe_load(f)

        if "model" in model_cfg:
            config.vlm_name = model_cfg["model"].get("name", config.vlm_name)
        if "memory" in model_cfg:
            mem = model_cfg["memory"]
            config.num_probes = mem.get("num_probes", config.num_probes)
            config.sampler_layers = mem.get("sampler_layers", config.sampler_layers)
            config.sampler_heads = mem.get("sampler_heads", config.sampler_heads)

        if "lora" in stage_cfg:
            lora = stage_cfg["lora"]
            config.lora_rank = lora.get("rank", config.lora_rank)
            config.lora_alpha = lora.get("alpha", config.lora_alpha)
            config.lora_dropout = lora.get("dropout", config.lora_dropout)

        return config


class MedSynapseV(nn.Module):
    """
    MedSynapse-V: Latent Diagnostic Memory Evolution framework.

    Supports three operating modes corresponding to training stages:
      - stage1: VLM frozen, P_phi trainable, next-token prediction warmup
      - stage2: VLM LoRA trainable, P_phi frozen, GRPO with causal reward
      - stage3: A_psi trainable, dual-branch JSD distillation

    At inference, only the VLM backbone + LoRA + A_psi are used;
    the anatomical encoder is entirely removed.
    """

    def __init__(self, config: MedSynapseVConfig):
        super().__init__()
        self.config = config
        self.vlm = None
        self.processor = None
        self.encoder = None
        self.memory_sampler = None
        self.autonomous_module = None
        self.injector = None
        self.vision_extractor = None
        self._stage = None

    def build(self, stage: int = 1, load_encoder: bool = True):
        """
        Build the model for a specific training stage.

        Args:
            stage: training stage (1, 2, 3, or 0 for inference)
            load_encoder: whether to load the anatomical encoder
        """
        self._stage = stage
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.vlm_dtype, torch.bfloat16)

        logger.info(f"Loading VLM: {self.config.vlm_name}")
        self.vlm = AutoModelForCausalLM.from_pretrained(
            self.config.vlm_name,
            torch_dtype=torch_dtype,
            attn_implementation=self.config.vlm_attn_impl,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.config.vlm_name,
            trust_remote_code=True,
        )

        d_hidden = self.vlm.config.hidden_size

        # Memory Sampler P_phi
        logger.info("Building Diagnostic Memory Sampler P_phi")
        self.memory_sampler = DiagnosticMemorySampler(
            num_probes=self.config.num_probes,
            d_encoder=self.config.encoder_feature_dim,
            d_hidden=d_hidden,
            num_layers=self.config.sampler_layers,
            num_heads=self.config.sampler_heads,
            ffn_dim=self.config.sampler_ffn_dim,
            probe_init_std=self.config.probe_init_std,
        )

        # Hidden Stream Injector
        self.injector = HiddenStreamInjector(
            num_memory_tokens=self.config.num_probes,
            d_hidden=d_hidden,
        )

        # Anatomical Encoder
        if load_encoder:
            logger.info("Loading MedSAM3 anatomical encoder (frozen)")
            self.encoder = MedSAMWrapper(
                pretrained_path=self.config.encoder_path,
                feature_dim=self.config.encoder_feature_dim,
                mask_threshold=self.config.mask_threshold,
                target_spatial=self.config.encoder_spatial,
            )

        # Stage-specific setup
        if stage == 1:
            self._setup_stage1()
        elif stage == 2:
            self._setup_stage2()
        elif stage == 3:
            self._setup_stage3(d_hidden)
        elif stage == 0:
            self._setup_inference(d_hidden)

        self._log_trainable_params()
        return self

    def _setup_stage1(self):
        """Stage I: freeze VLM + encoder, train P_phi only."""
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.memory_sampler.requires_grad_(True)

    def _setup_stage2(self):
        """Stage II: freeze P_phi + encoder, apply LoRA to VLM."""
        self.memory_sampler.requires_grad_(False)

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)
        logger.info(f"LoRA applied: r={self.config.lora_rank}, alpha={self.config.lora_alpha}")

    def _setup_stage3(self, d_hidden: int):
        """Stage III: freeze everything except A_psi."""
        # Freeze VLM (with LoRA merged or kept)
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.memory_sampler.requires_grad_(False)

        # Build Autonomous Memory Module
        logger.info("Building Autonomous Memory Module A_psi")
        self.autonomous_module = AutonomousMemoryModule(
            d_input=d_hidden,
            d_hidden=self.config.auto_hidden_dim,
            d_output=d_hidden,
            num_tokens=self.config.num_probes,
            num_layers=self.config.auto_num_layers,
        )

        # Vision feature extractor for A_psi input
        self.vision_extractor = QwenVisionFeatureExtractor(
            model=self.vlm,
            hidden_dim=d_hidden,
        )

    def _setup_inference(self, d_hidden: int):
        """Inference mode: VLM + LoRA + A_psi only, no encoder."""
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.memory_sampler = None
        self.encoder = None

        if self.autonomous_module is None:
            self.autonomous_module = AutonomousMemoryModule(
                d_input=d_hidden,
                d_hidden=self.config.auto_hidden_dim,
                d_output=d_hidden,
                num_tokens=self.config.num_probes,
                num_layers=self.config.auto_num_layers,
            )
        self.autonomous_module.requires_grad_(False)

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Stage {self._stage} | Total: {total/1e6:.1f}M | "
            f"Trainable: {trainable/1e6:.1f}M ({100*trainable/total:.2f}%)"
        )

    def generate_memory_privileged(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Teacher branch: generate M_pri via anatomical encoder + P_phi.
        Used in Stage I, II, and as teacher in Stage III.
        """
        assert self.encoder is not None, "Encoder required for privileged memory"
        enc_out = self.encoder(images)
        features = enc_out["features"]
        memory = self.memory_sampler(features)
        return {
            "memory": memory,
            "features": features,
            "masks": enc_out.get("masks"),
            "masks_flat": enc_out.get("masks_flat"),
        }

    def generate_memory_autonomous(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Student branch: generate M_auto via A_psi from VLM visual features.
        Used in Stage III training and at inference.
        """
        assert self.autonomous_module is not None, "A_psi required for autonomous memory"
        visual_features = self.vision_extractor(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        memory = self.autonomous_module(visual_features)
        return memory

    def generate_memory_counterfactual(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate intervened memory M' for causal counterfactual reward.
        M' = P_phi(E_ana(X) ⊙ B_bar)
        """
        assert self.encoder is not None
        enc_out = self.encoder(images)
        features_intervened = self.encoder.construct_counterfactual(
            enc_out["features"], enc_out["masks_flat"]
        )
        memory_intervened = self.memory_sampler(features_intervened)
        return memory_intervened

    def forward(self, **kwargs):
        """Dispatch to stage-specific forward logic."""
        raise NotImplementedError(
            "Use stage-specific engine classes for forward passes. "
            "This module serves as a component container."
        )
