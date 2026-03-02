"""
CLI: Decode JSON manga representations back to images.

Usage:
    python scripts/decode.py \
        --model-dir checkpoints/infiller/phase2 \
        --json-dir /path/to/json \
        --output-dir /path/to/output
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.decode")


def main():
    parser = argparse.ArgumentParser(description="Decode JSON to manga images")
    parser.add_argument("--model-dir", required=True,
                        help="Directory with trained infiller checkpoints")
    parser.add_argument("--json-dir", required=True,
                        help="Directory containing MangaPage JSON files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save rendered images")
    parser.add_argument("--sd-model", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of SD denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    args = parser.parse_args()

    from mangaka.pipeline.decode import decode_directory

    paths = decode_directory(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        sd_model=args.sd_model,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
    )
    logger.info("Decoded %d pages", len(paths))


if __name__ == "__main__":
    main()
