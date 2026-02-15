"""
CLI: Run the distillation loop.

Usage:
    python scripts/distill.py \
        --unlabelled-dir /path/to/unlabelled/manga \
        --annotation-dir /path/to/annotations \
        --detector-path checkpoints/detector.safetensors \
        --config configs/distill.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.distill")


def main():
    parser = argparse.ArgumentParser(description="Run distillation loop")
    parser.add_argument("--unlabelled-dir", required=True,
                        help="Directory with unlabelled manga images")
    parser.add_argument("--annotation-dir", required=True,
                        help="Directory for annotations (candidates + verified)")
    parser.add_argument("--detector-path", required=True,
                        help="Path to initial trained detector checkpoint")
    parser.add_argument("--config", default="configs/distill.yaml")
    parser.add_argument("--provider", choices=["claude", "openai", "gemini"], default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    args = parser.parse_args()

    # Load config
    cfg = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    # Required paths
    cfg["unlabelled_dir"] = args.unlabelled_dir
    cfg["annotation_dir"] = args.annotation_dir
    cfg["detector_path"] = args.detector_path

    # CLI overrides
    if args.provider:
        cfg["provider"] = args.provider
    if args.max_rounds:
        cfg["max_rounds"] = args.max_rounds

    # Defaults
    cfg.setdefault("provider", "claude")
    cfg.setdefault("max_rounds", 5)
    cfg.setdefault("verification_sample", 0.2)
    cfg.setdefault("max_concurrency", 10)
    cfg.setdefault("retrain_epochs", 20)
    cfg.setdefault("retrain_lr", 1e-4)
    cfg.setdefault("output_path", "checkpoints/detector_distilled.safetensors")

    logger.info("Distillation config: %s", cfg)

    from mangaka.detector.distill import run_distillation
    asyncio.run(run_distillation(cfg))


if __name__ == "__main__":
    main()
