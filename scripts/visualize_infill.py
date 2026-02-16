"""
Visualize original vs infilled manga page side-by-side.

Usage:
    python scripts/visualize_infill.py --page-id 0015
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mangaka.schema import MangaPage
from mangaka.infiller.model import MangaInfiller, LayoutControlNet
from mangaka.pipeline.decode import decode_page
from mangaka.utils.viz import draw_annotations, side_by_side

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize original vs infilled manga page")
    parser.add_argument("--page-id", type=str, default="0015", help="Page ID (e.g. 0015)")
    parser.add_argument("--image-dir", type=str,
                        default="/home/ubuntu/Code/manga_translate/manga_translate/yolo/images")
    parser.add_argument("--annotation-dir", type=str, default="data/manga_dataset/pages")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/infiller/phase2/controlnet_best.pt")
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30, help="Inference steps (fewer = faster)")
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--output-dir", type=str, default="output/visualize")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Path to LoRA adapter weights (e.g. checkpoints/infiller/phase2/unet_lora)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load original image
    image_path = Path(args.image_dir) / f"{args.page_id}.jpg"
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return
    original = Image.open(image_path).convert("RGB")
    logger.info("Loaded original image: %s (%dx%d)", image_path.name, *original.size)

    # Load annotation
    annotation_path = Path(args.annotation_dir) / f"{args.page_id}.json"
    if not annotation_path.exists():
        logger.error("Annotation not found: %s", annotation_path)
        return
    page = MangaPage.from_json(annotation_path)
    logger.info("Loaded annotation: %d panels, %d elements", page.num_panels, page.num_elements)

    # Load ControlNet checkpoint
    checkpoint_path = Path(args.checkpoint)
    controlnet = LayoutControlNet()
    if checkpoint_path.exists():
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        controlnet.load_state_dict(state_dict)
        logger.info("Loaded ControlNet from %s", checkpoint_path)
    else:
        logger.warning("No checkpoint found at %s, using untrained ControlNet", checkpoint_path)

    controlnet = controlnet.to(device).half()

    # Build infiller
    infiller = MangaInfiller(
        sd_model_id=args.sd_model,
        resolution=args.resolution,
        controlnet=controlnet,
    )
    infiller.load_sd_pipeline(device)

    # Load LoRA if provided
    if args.lora_path:
        lora_path = Path(args.lora_path)
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            infiller.load_lora_weights(lora_path)
            logger.info("Loaded LoRA from %s", lora_path)
        else:
            logger.warning("LoRA path %s not found or missing adapter_config.json", lora_path)

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    # Render the page
    logger.info("Rendering page %s (%d steps, guidance=%.1f)...", args.page_id, args.steps, args.guidance)
    generated = decode_page(
        page, infiller,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
    )

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Original with annotations overlaid
    annotated = draw_annotations(original, page)
    annotated.save(output_dir / f"{args.page_id}_annotated.png")
    logger.info("Saved annotated original")

    # 2. Generated image
    generated.save(output_dir / f"{args.page_id}_generated.png")
    logger.info("Saved generated image")

    # 3. Side-by-side comparison
    side_by_side(
        original, generated,
        output_dir / f"{args.page_id}_comparison.png",
        label_left="Original",
        label_right="Infilled",
    )
    logger.info("Saved side-by-side comparison to %s", output_dir / f"{args.page_id}_comparison.png")

    print(f"\nDone! Outputs saved to {output_dir}/")
    print(f"  - {args.page_id}_annotated.png   (original + bboxes)")
    print(f"  - {args.page_id}_generated.png   (infilled from JSON)")
    print(f"  - {args.page_id}_comparison.png  (side-by-side)")


if __name__ == "__main__":
    main()
