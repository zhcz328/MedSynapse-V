"""
Pre-compute MedSAM3 features and masks for training acceleration.

Usage:
    python scripts/cache_medsam_features.py \
        --encoder_path checkpoints/medsam3_vit_b.pth \
        --data_config configs/datasets/stage2_rl_mixed.yaml \
        --output_dir cache/medsam_features
"""

import os, sys, argparse, json, torch, yaml
from torchvision import transforms
from tqdm import tqdm
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.encoders.medsam_wrapper import MedSAMWrapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--encoder_path", type=str, required=True)
    p.add_argument("--data_config", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="cache/medsam_features")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mask_threshold", type=float, default=0.7)
    p.add_argument("--feature_dim", type=int, default=1024)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    encoder = MedSAMWrapper(
        pretrained_path=args.encoder_path,
        feature_dim=args.feature_dim,
        mask_threshold=args.mask_threshold,
    ).to(args.device)

    from data.datasets.slake_pathvqa import MixedRLDataset
    dataset = MixedRLDataset(omnimedvqa_samples=3000, slake_samples=500, pathvqa_samples=500)
    logger.info(f"Caching {len(dataset)} samples")

    feat_dir = os.path.join(args.output_dir, "features")
    mask_dir = os.path.join(args.output_dir, "masks")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    tfm = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])

    for idx in tqdm(range(len(dataset)), desc="Caching"):
        image = dataset[idx].get("image")
        if image is None:
            continue
        img_t = tfm(image).unsqueeze(0).to(args.device)
        with torch.no_grad():
            out = encoder(img_t)
        torch.save(out["features"].squeeze(0).cpu(), os.path.join(feat_dir, f"{idx:06d}.pt"))
        torch.save(out["masks_flat"].squeeze(0).cpu(), os.path.join(mask_dir, f"{idx:06d}.pt"))

    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump({"num_samples": len(dataset), "feature_dim": args.feature_dim,
                    "mask_threshold": args.mask_threshold}, f, indent=2)
    logger.info(f"Done: {args.output_dir}")


if __name__ == "__main__":
    main()
