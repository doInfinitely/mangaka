#!/usr/bin/env python3
"""Fetch artist images from ExHentai for style transfer training.

Usage:
    python scripts/fetch_style_data.py \
        --artists "artist_a,artist_b" \
        [--cache-dir data/gallery_cache/] \
        [--image-dir data/style_images/] \
        [--config configs/gallery.yaml] \
        [--build-bank]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys

import yaml

from mangaka.gallery import ExHentaiClient, GalleryCache, fetch_artist_images

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fetch artist images for style transfer")
    parser.add_argument(
        "--artists",
        required=True,
        help="Comma-separated artist names",
    )
    parser.add_argument("--cache-dir", default=None, help="Gallery cache directory")
    parser.add_argument("--image-dir", default="data/style_images", help="Output image directory")
    parser.add_argument("--config", default="configs/gallery.yaml", help="Config YAML")
    parser.add_argument(
        "--build-bank",
        action="store_true",
        help="Also run build_style_bank after fetching",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cache_dir = args.cache_dir or cfg.get("cache_dir", "data/gallery_cache")
    artist_names = [a.strip() for a in args.artists.split(",") if a.strip()]

    if not artist_names:
        logger.error("No artist names provided")
        sys.exit(1)

    client = ExHentaiClient(
        rate_limit=cfg.get("rate_limit", 2.0),
        max_retries=cfg.get("max_retries", 3),
        timeout=cfg.get("timeout", 30.0),
    )
    cache = GalleryCache(cache_dir)

    result = fetch_artist_images(
        artist_names=artist_names,
        client=client,
        cache=cache,
        output_dir=args.image_dir,
        max_galleries_per_artist=cfg.get("max_galleries_per_artist", 20),
    )

    for artist, paths in result.items():
        logger.info("%s: %d images", artist, len(paths))

    if args.build_bank:
        logger.info("Running build_style_bank...")
        subprocess.run(
            [
                sys.executable,
                "scripts/build_style_bank.py",
                "--image-dir",
                args.image_dir,
                "--bank-dir",
                "data/style_bank",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
