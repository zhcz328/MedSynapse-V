import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml

from core.encoders.medsam_wrapper import MedSAMWrapper
from core.encoders.qwen_vision import QwenVisionFeatureExtractor
from core.memory import AutonomousMemoryModule, DiagnosticMemorySampler, HiddenStreamInjector

logger = logging.getLogger(__name__)


@dataclass
class MedSynapseVConfig:
    model_type: str = "qwen3_vl"
    vlm_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    vlm_dtype: str = "bfloat16"
    vlm_attn_impl: Optional[str] = "flash_attention_2"
    device_map: Optional[str] = None

    encoder_path: str = "checkpoints/medsam_vit_b.pth"
    encoder_img_size: int = 1024
    encoder_patch_size: int = 16
    encoder_embed_dim: int = 768
    encoder_depth: int = 12
    encoder_num_heads: int = 12
    encoder_mlp_ratio: float = 4.0
    encoder_out_chans: int = 256
    encoder_backend: str = "auto"
    encoder_feature_backend: Optional[str] = None
    sam3_model_name: str = "facebook/sam3"
    sam3_cache_dir: Optional[str] = None
    encoder_feature_dim: int = 1024
    encoder_spatial: Tuple[int, int] = (64, 64)
    mask_threshold: float = 0.7
    medsam3_config_path: Optional[str] = None
    medsam3_lora_path: Optional[str] = None
    medsam3_repo_dir: Optional[str] = None
    medsam3_script_path: Optional[str] = None
    medsam3_prompt: str = "lesion"
    medsam3_threshold: float = 0.5
    medsam3_nms_iou: float = 0.5
    medsam3_resolution: int = 1024
    medsam3_cache_dir: str = "cache/medsam3_masks"
    medsam3_python: str = "python3"
    medsam3_timeout_s: int = 300

    num_probes: int = 16
    sampler_layers: int = 2
    sampler_heads: int = 8
    sampler_ffn_dim: int = 4096
    probe_init_std: float = 0.02

    auto_hidden_dim: int = 4096
    auto_num_layers: int = 2

    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )

    @classmethod
    def from_yaml(
        cls,
        model_config_path: str,
        stage_config_path: Optional[str] = None,
        encoder_config_path: Optional[str] = None,
    ) -> "MedSynapseVConfig":
        config = cls()
        model_cfg = _load_yaml(model_config_path)
        stage_cfg = _load_yaml(stage_config_path) if stage_config_path else {}
        encoder_cfg = _load_yaml(encoder_config_path) if encoder_config_path else {}

        model = model_cfg.get("model", {})
        config.vlm_name = model.get("name", config.vlm_name)
        config.model_type = model.get("type", config.model_type)

        loading = model_cfg.get("loading", {})
        config.vlm_dtype = loading.get("torch_dtype", config.vlm_dtype)
        config.vlm_attn_impl = loading.get("attn_implementation", config.vlm_attn_impl)
        config.device_map = loading.get("device_map", config.device_map)

        memory = model_cfg.get("memory", {})
        config.num_probes = memory.get("num_probes", config.num_probes)
        config.sampler_layers = memory.get("sampler_layers", config.sampler_layers)
        config.sampler_heads = memory.get("sampler_heads", config.sampler_heads)
        config.sampler_ffn_dim = memory.get("ffn_dim", config.sampler_ffn_dim)
        config.probe_init_std = memory.get("probe_init_std", config.probe_init_std)

        autonomous = model_cfg.get("autonomous", {})
        config.auto_hidden_dim = autonomous.get("hidden_dim", config.auto_hidden_dim)
        config.auto_num_layers = autonomous.get("num_layers", config.auto_num_layers)

        enc = encoder_cfg.get("encoder", {})
        if enc:
            config.encoder_path = enc.get("pretrained_path", config.encoder_path)
            arch = enc.get("architecture", {})
            config.encoder_img_size = arch.get("image_size", config.encoder_img_size)
            config.encoder_patch_size = arch.get("patch_size", config.encoder_patch_size)
            config.encoder_embed_dim = arch.get("embed_dim", config.encoder_embed_dim)
            config.encoder_depth = arch.get("depth", config.encoder_depth)
            config.encoder_num_heads = arch.get("num_heads", config.encoder_num_heads)
            config.encoder_mlp_ratio = arch.get("mlp_ratio", config.encoder_mlp_ratio)
            config.encoder_out_chans = arch.get("out_chans", config.encoder_out_chans)
            config.encoder_backend = enc.get("backend", config.encoder_backend)
            config.encoder_feature_backend = enc.get("feature_backend", config.encoder_feature_backend)
            sam3_cfg = enc.get("sam3", {})
            config.sam3_model_name = sam3_cfg.get("model_name", config.sam3_model_name)
            config.sam3_cache_dir = sam3_cfg.get("cache_dir", config.sam3_cache_dir)
            feat = enc.get("feature_extraction", {})
            config.encoder_feature_dim = feat.get("output_channels", config.encoder_feature_dim)
            config.encoder_spatial = tuple(feat.get("output_spatial", config.encoder_spatial))
            seg = enc.get("segmentation_head", {})
            config.mask_threshold = seg.get("mask_threshold", config.mask_threshold)
            medsam3 = enc.get("medsam3", {})
            config.medsam3_config_path = medsam3.get("config_path", config.medsam3_config_path)
            config.medsam3_lora_path = medsam3.get("lora_path", config.medsam3_lora_path)
            config.medsam3_repo_dir = medsam3.get("repo_dir", config.medsam3_repo_dir)
            config.medsam3_script_path = medsam3.get("script_path", config.medsam3_script_path)
            config.medsam3_prompt = medsam3.get("prompt", config.medsam3_prompt)
            config.medsam3_threshold = medsam3.get("threshold", config.medsam3_threshold)
            config.medsam3_nms_iou = medsam3.get("nms_iou", config.medsam3_nms_iou)
            config.medsam3_resolution = medsam3.get("resolution", config.medsam3_resolution)
            config.medsam3_cache_dir = medsam3.get("cache_dir", config.medsam3_cache_dir)
            config.medsam3_python = medsam3.get("python", config.medsam3_python)
            config.medsam3_timeout_s = medsam3.get("timeout_s", config.medsam3_timeout_s)

        lora = stage_cfg.get("lora", {})
        config.lora_rank = lora.get("rank", config.lora_rank)
        config.lora_alpha = lora.get("alpha", config.lora_alpha)
        config.lora_dropout = lora.get("dropout", config.lora_dropout)
        config.lora_target_modules = tuple(
            lora.get("target_modules", config.lora_target_modules)
        )
        return config


class MedSynapseV(nn.Module):
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
        self._stage = stage
        self._build_vlm()
        d_hidden = self._hidden_size()

        self.memory_sampler = DiagnosticMemorySampler(
            num_probes=self.config.num_probes,
            d_encoder=self.config.encoder_feature_dim,
            d_hidden=d_hidden,
            num_layers=self.config.sampler_layers,
            num_heads=self.config.sampler_heads,
            ffn_dim=self.config.sampler_ffn_dim,
            probe_init_std=self.config.probe_init_std,
        )
        self.injector = HiddenStreamInjector(
            num_memory_tokens=self.config.num_probes,
            d_hidden=d_hidden,
        )

        if load_encoder:
            self.encoder = self._build_encoder()

        if stage == 1:
            self._setup_stage1()
        elif stage == 2:
            self._setup_stage2()
        elif stage == 3:
            self._setup_stage3(d_hidden)
        elif stage == 0:
            self._setup_inference(d_hidden)
        else:
            raise ValueError(f"Unsupported stage: {stage}")

        self._log_trainable_params()
        return self

    def _build_vlm(self):
        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("transformers>=4.57 is required for Qwen3-VL") from exc

        model_cls = _resolve_vl_model_class(transformers, self.config.model_type)
        kwargs = {
            "trust_remote_code": True,
        }
        dtype = _resolve_dtype(self.config.vlm_dtype)
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.config.vlm_attn_impl and self.config.vlm_attn_impl != "auto":
            kwargs["attn_implementation"] = self.config.vlm_attn_impl
        if self.config.device_map:
            kwargs["device_map"] = self.config.device_map

        logger.info("Loading VLM %s with %s", self.config.vlm_name, model_cls.__name__)
        self.vlm = model_cls.from_pretrained(self.config.vlm_name, **kwargs)
        self.processor = AutoProcessor.from_pretrained(
            self.config.vlm_name,
            trust_remote_code=True,
        )

    def _build_encoder(self):
        encoder_path = self.config.encoder_path
        needs_local_checkpoint = (
            (self.config.encoder_feature_backend or self.config.encoder_backend)
            not in {"sam3_hf", "sam3", "medsam3"}
        )
        if needs_local_checkpoint and (not encoder_path or not os.path.exists(encoder_path)):
            raise FileNotFoundError(
                "MedSAM checkpoint is required for privileged training: "
                f"{encoder_path}. Set configs/models/medsam_encoder_frozen.yaml:encoder.pretrained_path"
            )
        return MedSAMWrapper(
            pretrained_path=encoder_path,
            img_size=self.config.encoder_img_size,
            patch_size=self.config.encoder_patch_size,
            embed_dim=self.config.encoder_embed_dim,
            depth=self.config.encoder_depth,
            num_heads=self.config.encoder_num_heads,
            mlp_ratio=self.config.encoder_mlp_ratio,
            out_chans=self.config.encoder_out_chans,
            feature_dim=self.config.encoder_feature_dim,
            mask_threshold=self.config.mask_threshold,
            target_spatial=self.config.encoder_spatial,
            backend=self.config.encoder_backend,
            feature_backend=self.config.encoder_feature_backend,
            sam3_model_name=self.config.sam3_model_name,
            sam3_cache_dir=self.config.sam3_cache_dir,
            medsam3_config_path=self.config.medsam3_config_path,
            medsam3_lora_path=self.config.medsam3_lora_path,
            medsam3_repo_dir=self.config.medsam3_repo_dir,
            medsam3_script_path=self.config.medsam3_script_path,
            medsam3_prompt=self.config.medsam3_prompt,
            medsam3_threshold=self.config.medsam3_threshold,
            medsam3_nms_iou=self.config.medsam3_nms_iou,
            medsam3_resolution=self.config.medsam3_resolution,
            medsam3_cache_dir=self.config.medsam3_cache_dir,
            medsam3_python=self.config.medsam3_python,
            medsam3_timeout_s=self.config.medsam3_timeout_s,
        )

    def _setup_stage1(self):
        for param in self.vlm.parameters():
            param.requires_grad = False
        if self.encoder is not None:
            self.encoder.requires_grad_(False)
        self.memory_sampler.requires_grad_(True)

    def _setup_stage2(self):
        self.memory_sampler.requires_grad_(False)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)

        self._apply_lora(trainable=True)

    def _setup_stage3(self, d_hidden: int):
        self._apply_lora(trainable=False)
        try:
            self.vlm.enable_input_require_grads()
        except Exception:
            pass

        for param in self.vlm.parameters():
            param.requires_grad = False
        self.memory_sampler.requires_grad_(False)
        if self.encoder is not None:
            self.encoder.requires_grad_(False)

        self.autonomous_module = AutonomousMemoryModule(
            d_input=d_hidden,
            d_hidden=self.config.auto_hidden_dim,
            d_output=d_hidden,
            num_tokens=self.config.num_probes,
            num_layers=self.config.auto_num_layers,
        )
        self.vision_extractor = QwenVisionFeatureExtractor(self.vlm, hidden_dim=d_hidden)

    def _setup_inference(self, d_hidden: int):
        self._apply_lora(trainable=False)
        for param in self.vlm.parameters():
            param.requires_grad = False
        self.memory_sampler = None
        self.encoder = None
        self.autonomous_module = AutonomousMemoryModule(
            d_input=d_hidden,
            d_hidden=self.config.auto_hidden_dim,
            d_output=d_hidden,
            num_tokens=self.config.num_probes,
            num_layers=self.config.auto_num_layers,
        )
        self.autonomous_module.requires_grad_(False)
        self.vision_extractor = QwenVisionFeatureExtractor(self.vlm, hidden_dim=d_hidden)

    def _apply_lora(self, trainable: bool):
        if hasattr(self.vlm, "peft_config"):
            self._set_lora_trainable(trainable)
            return
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("peft is required for MedSynapse-V LoRA adapters") from exc

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=list(self.config.lora_target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )
        self.vlm = get_peft_model(self.vlm, lora_config)
        self._set_lora_trainable(trainable)
        logger.info("LoRA applied: r=%s alpha=%s", self.config.lora_rank, self.config.lora_alpha)

    def _set_lora_trainable(self, trainable: bool):
        for name, param in self.vlm.named_parameters():
            is_adapter = "lora_" in name or "modules_to_save" in name
            param.requires_grad = bool(trainable and is_adapter)

    def _log_trainable_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ratio = 100.0 * trainable / max(total, 1)
        logger.info(
            "Stage %s | Total %.2fM | Trainable %.2fM (%.2f%%)",
            self._stage,
            total / 1e6,
            trainable / 1e6,
            ratio,
        )

    def generate_memory_privileged(
        self,
        images: torch.Tensor,
        concept_prompts=None,
    ) -> Dict[str, torch.Tensor]:
        if self.encoder is None:
            raise RuntimeError("Encoder required for privileged memory")
        enc_out = self.encoder(images, concept_prompts=concept_prompts)
        memory = self.memory_sampler(enc_out["features"])
        return {
            "memory": memory,
            "features": enc_out["features"],
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
        if self.autonomous_module is None:
            raise RuntimeError("Autonomous memory module is not built")
        if self.vision_extractor is None:
            self.vision_extractor = QwenVisionFeatureExtractor(
                self.vlm,
                hidden_dim=self._hidden_size(),
            )
        device = input_ids.device
        if pixel_values is not None:
            pixel_values = pixel_values.to(device)
        if image_grid_thw is not None and isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        visual_features = self.vision_extractor(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        return self.autonomous_module(visual_features)

    def generate_memory_counterfactual(
        self,
        images: torch.Tensor,
        concept_prompts=None,
    ) -> torch.Tensor:
        if self.encoder is None:
            raise RuntimeError("Encoder required for counterfactual memory")
        enc_out = self.encoder(images, concept_prompts=concept_prompts)
        features_intervened = self.encoder.construct_counterfactual(
            enc_out["features"],
            enc_out["masks_flat"],
        )
        return self.memory_sampler(features_intervened)

    def forward(self, **kwargs):
        raise NotImplementedError("Use the stage engines for training/evaluation")

    def _hidden_size(self) -> int:
        candidates = [getattr(self.vlm, "config", None)]
        for obj in list(candidates):
            if obj is not None:
                candidates.append(getattr(obj, "text_config", None))
        base_model = getattr(self.vlm, "base_model", None)
        if base_model is not None:
            candidates.append(getattr(base_model, "config", None))

        for obj in candidates:
            if obj is None:
                continue
            for attr in ("hidden_size", "hidden_dim", "d_model"):
                value = getattr(obj, attr, None)
                if value is not None:
                    return int(value)
        raise RuntimeError("Unable to resolve VLM hidden size from model config")


def _resolve_vl_model_class(transformers_module, model_type: str):
    preferred = []
    if model_type in {"qwen3_vl", "qwen3-vl"}:
        preferred.extend(["Qwen3VLForConditionalGeneration"])
    if model_type in {"qwen2_5_vl", "qwen2.5-vl", "qwen2_5-vl"}:
        preferred.extend(["Qwen2_5_VLForConditionalGeneration"])
    preferred.extend(
        [
            "AutoModelForImageTextToText",
            "AutoModelForVision2Seq",
            "AutoModelForCausalLM",
        ]
    )
    for name in preferred:
        cls = getattr(transformers_module, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "No compatible Qwen-VL model class found. Install a recent transformers build."
    )


def _resolve_dtype(dtype_name: str):
    if dtype_name in {None, "auto"}:
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }.get(dtype_name, dtype_name)


def _load_yaml(path: Optional[str]) -> Dict:
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}
