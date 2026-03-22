#!/usr/bin/env python3
"""Precompute style embeddings for all artists and save to disk.

Walks an image directory organised by artist, encodes every image through
the StyleEncoder, and saves one .pt file per artist in the bank directory.

Usage:
    python scripts/build_style_bank.py \
        --image-dir data/images/ \
        --bank-dir data/style_bank/ \
        [--config configs/style.yaml] \
        [--device cuda]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from mangaka.style.encoder import StyleEncoder
from mangaka.style.bank import StyleBank

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def build_bank(
    image_dir: str | Path,
    bank_dir: str | Path,
    resolution: int = 512,
    batch_size: int = 32,
    style_dim: int = 512,
    device: str = "cuda",
):
    """Encode all artist images and save to the style bank."""
    image_dir = Path(image_dir)
    bank = StyleBank(bank_dir, style_dim=style_dim)
    encoder = StyleEncoder(style_dim=style_dim).to(device)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),  # [0, 1]
    ])

    artist_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())
    logger.info("Found %d artist directories", len(artist_dirs))

    for artist_dir in tqdm(artist_dirs, desc="Artists"):
        artist_id = artist_dir.name
        images = sorted(
            p for p in artist_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            logger.warning("No images for artist '%s', skipping", artist_id)
            continue

        all_embeddings = []

        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_paths = images[i : i + batch_size]
            batch_tensors = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_tensors.append(transform(img))
                except Exception as e:
                    logger.warning("Failed to load %s: %s", img_path, e)
                    # Placeholder zero embedding to keep indices aligned
                    batch_tensors.append(torch.zeros(3, resolution, resolution))

            batch = torch.stack(batch_tensors).to(device)
            embeddings = encoder.encode_batch(batch)
            all_embeddings.append(embeddings.cpu())

        artist_embeddings = torch.cat(all_embeddings, dim=0)
        bank.save_artist(artist_id, artist_embeddings)

    logger.info("Style bank built: %d artists", len(artist_dirs))


def main():
    parser = argparse.ArgumentParser(description="Build style embedding bank")
    parser.add_argument("--image-dir", required=True, help="Root directory with artist subdirs")
    parser.add_argument("--bank-dir", required=True, help="Output directory for .pt files")
    parser.add_argument("--config", default="configs/style.yaml", help="Config YAML")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_bank(
        image_dir=args.image_dir,
        bank_dir=args.bank_dir,
        resolution=cfg.get("resolution", 512),
        batch_size=cfg.get("encoder", {}).get("batch_size", 32),
        style_dim=cfg.get("style_dim", 512),
        device=args.device,
    )


if __name__ == "__main__":
    main()
