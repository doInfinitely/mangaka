"""
CLI: Train the MangaInfiller (SD inpainting + ControlNet).

Usage:
    accelerate launch scripts/train_infiller.py \
        --image-dir /path/to/manga \
        --annotation-dir /path/to/annotations \
        --config configs/infiller.yaml
"""

from __future__ import annotations

import argparse
import logging
import os

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.train_infiller")


def main():
    parser = argparse.ArgumentParser(description="Train MangaInfiller")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--annotation-dir", required=True)
    parser.add_argument("--config", default="configs/infiller.yaml")

    # CLI overrides
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # CLI overrides
    for key in ("epochs", "batch_size", "resolution", "save_path"):
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    # Defaults
    cfg.setdefault("sd_model", "stabilityai/stable-diffusion-2-inpainting")
    cfg.setdefault("resolution", 512)
    cfg.setdefault("epochs", 100)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("controlnet_lr", 1e-4)
    cfg.setdefault("unfreeze_sd_lr", 1e-6)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("freeze_sd_epochs", 20)
    cfg.setdefault("warmup_steps", 500)
    cfg.setdefault("val_split", 0.1)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("amp", True)
    cfg.setdefault("gradient_accumulation_steps", 4)
    cfg.setdefault("seed", 42)
    cfg.setdefault("save_path", "checkpoints/infiller")

    logger.info("Config: %s", cfg)

    from mangaka.infiller.train import train_infiller
    train_infiller(args.image_dir, args.annotation_dir, cfg)


if __name__ == "__main__":
    main()
