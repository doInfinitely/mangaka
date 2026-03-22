"""
CLI: Encode manga images to JSON using the trained detector.

Usage:
    python scripts/encode.py \
        --model-path checkpoints/detector.safetensors \
        --image-dir /path/to/manga \
        --output-dir /path/to/json
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.encode")


def main():
    parser = argparse.ArgumentParser(description="Encode manga images to JSON")
    parser.add_argument("--model-path", required=True,
                        help="Path to trained detector checkpoint")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing manga images")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save JSON annotations")
    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--lm-model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-detections", type=int, default=50)
    args = parser.parse_args()

    from mangaka.pipeline.encode import encode_directory

    paths = encode_directory(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        backbone=args.backbone,
        lm_model=args.lm_model,
        max_detections=args.max_detections,
    )
    logger.info("Encoded %d pages", len(paths))


if __name__ == "__main__":
    main()
