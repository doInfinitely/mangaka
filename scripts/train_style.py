#!/usr/bin/env python3
"""Train the style transfer GAN."""

from __future__ import annotations

import argparse
import logging

import yaml

from mangaka.style.train import train_style_gan


def main():
    parser = argparse.ArgumentParser(description="Train style transfer GAN")
    parser.add_argument("--image-dir", required=True, help="Path to artist image directories")
    parser.add_argument("--bank-dir", required=True, help="Path to precomputed style bank .pt files")
    parser.add_argument("--config", default="configs/style.yaml", help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_style_gan(
        image_dir=args.image_dir,
        bank_dir=args.bank_dir,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
