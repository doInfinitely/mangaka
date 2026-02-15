"""
CLI: Build a JSON manga dataset for RAG and LLM finetuning.

Usage (from existing annotations):
    python scripts/build_dataset.py \
        --annotation-dir /path/to/annotations \
        --image-dir /path/to/manga \
        --output-dir /path/to/dataset

Usage (generate with detector):
    python scripts/build_dataset.py \
        --model-path checkpoints/detector.safetensors \
        --image-dir /path/to/manga \
        --output-dir /path/to/dataset
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.build_dataset")


def main():
    parser = argparse.ArgumentParser(description="Build JSON manga dataset")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing manga images")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write the JSON dataset")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--annotation-dir",
                       help="Use existing annotations from this directory")
    group.add_argument("--model-path",
                       help="Generate annotations with this detector checkpoint")

    parser.add_argument("--backbone", default="resnet50",
                        choices=["resnet18", "resnet34", "resnet50"])
    parser.add_argument("--gpt2-model", default="gpt2")
    parser.add_argument("--max-detections", type=int, default=50)
    args = parser.parse_args()

    from mangaka.pipeline.build_dataset import build_dataset

    index = build_dataset(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        annotation_dir=args.annotation_dir,
        backbone=args.backbone,
        gpt2_model=args.gpt2_model,
        max_detections=args.max_detections,
    )
    logger.info("Dataset built: %d pages", index["total_pages"])


if __name__ == "__main__":
    main()
