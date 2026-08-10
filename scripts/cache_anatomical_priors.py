import argparse
import json
import logging
import os
import sys

import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.encoders.medsam_wrapper import MedSAMWrapper
from data.loader import build_dataset_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def cache_features(
    encoder_config,
    data_config,
    output_dir: str,
    model_config=None,
    device: str = None,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    enc = encoder_config.get("encoder", {})
    arch = enc.get("architecture", {})
    feat = enc.get("feature_extraction", {})
    seg = enc.get("segmentation_head", {})
    path = enc.get("pretrained_path")
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"MedSAM checkpoint is required for feature caching: {path}"
        )

    encoder = MedSAMWrapper(
        pretrained_path=path,
        img_size=arch.get("image_size", 1024),
        patch_size=arch.get("patch_size", 16),
        embed_dim=arch.get("embed_dim", 768),
        depth=arch.get("depth", 12),
        num_heads=arch.get("num_heads", 12),
        mlp_ratio=arch.get("mlp_ratio", 4.0),
        out_chans=arch.get("out_chans", 256),
        feature_dim=feat.get("output_channels", 1024),
        target_spatial=tuple(feat.get("output_spatial", [64, 64])),
        mask_threshold=seg.get("mask_threshold", 0.7),
        backend=enc.get("backend", "auto"),
        medsam3_config_path=enc.get("medsam3", {}).get("config_path"),
        medsam3_lora_path=enc.get("medsam3", {}).get("lora_path"),
        medsam3_repo_dir=enc.get("medsam3", {}).get("repo_dir"),
        medsam3_script_path=enc.get("medsam3", {}).get("script_path"),
        medsam3_prompt=enc.get("medsam3", {}).get("prompt", "lesion"),
        medsam3_threshold=enc.get("medsam3", {}).get("threshold", 0.5),
        medsam3_nms_iou=enc.get("medsam3", {}).get("nms_iou", 0.5),
        medsam3_resolution=enc.get("medsam3", {}).get("resolution", 1024),
        medsam3_cache_dir=enc.get("medsam3", {}).get("cache_dir", "cache/medsam3_masks"),
        medsam3_python=enc.get("medsam3", {}).get("python", "python3"),
        medsam3_timeout_s=enc.get("medsam3", {}).get("timeout_s", 300),
    ).to(device)

    dataset = build_dataset_from_config(data_config)
    feat_dir = os.path.join(output_dir, "features")
    mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    target_size = int(enc.get("preprocessing", {}).get("target_size", arch.get("image_size", 1024)))
    for idx in tqdm(range(len(dataset)), desc="Caching MedSAM features"):
        sample = dataset[idx]
        image = sample.image.convert("RGB").resize((target_size, target_size))
        image_t = _pil_to_tensor_255(image).unsqueeze(0).to(device)
        prompt = getattr(sample, "concept_prompt", None)
        with torch.no_grad():
            out = encoder(image_t, concept_prompts=[prompt] if prompt else None)
        torch.save(out["features"].squeeze(0).cpu(), os.path.join(feat_dir, f"{idx:06d}.pt"))
        torch.save(out["masks_flat"].squeeze(0).cpu(), os.path.join(mask_dir, f"{idx:06d}.pt"))

    metadata = {
        "num_samples": len(dataset),
        "feature_dim": int(out["features"].shape[-1]) if len(dataset) else None,
        "encoder_checkpoint": path,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Cached features: %s", output_dir)


def _pil_to_tensor_255(image):
    from torchvision.transforms import functional as TF

    return TF.to_tensor(image) * 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--data_config", type=str, required=True)
    parser.add_argument("--model_config", type=str, default="configs/models/qwen3_vl_8b.yaml")
    parser.add_argument("--encoder_config", type=str, default="configs/models/medsam_encoder_frozen.yaml")
    parser.add_argument("--output_dir", type=str, default="cache/medsam_features")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    with open(args.data_config) as f:
        data_config = yaml.safe_load(f) or {}
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f) or {}
    with open(args.encoder_config) as f:
        encoder_config = yaml.safe_load(f) or {}
    if args.encoder_path:
        encoder_config.setdefault("encoder", {})["pretrained_path"] = args.encoder_path

    cache_features(
        encoder_config=encoder_config,
        data_config=data_config,
        output_dir=args.output_dir,
        model_config=model_config,
        device=args.device,
    )


if __name__ == "__main__":
    main()
