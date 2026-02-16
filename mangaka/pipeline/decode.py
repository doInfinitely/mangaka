"""
Decode pipeline: MangaPage JSON → manga image using the trained infiller.

Hierarchical rendering:
  1. Start with blank canvas
  2. Render each panel using SD inpainting conditioned on panel description
  3. Render each element within panels conditioned on element description
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image

from mangaka.schema import MangaPage
from mangaka.infiller.model import MangaInfiller, LayoutControlNet

logger = logging.getLogger(__name__)


def decode_page(
    page: MangaPage,
    infiller: MangaInfiller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> Image.Image:
    """Render a MangaPage JSON into a manga image.

    Args:
        page: structured manga page representation
        infiller: trained MangaInfiller
        num_inference_steps: SD denoising steps
        guidance_scale: classifier-free guidance scale

    Returns:
        PIL Image of the rendered manga page
    """
    return infiller.render_page(
        page,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )


def decode_directory(
    json_dir: str,
    output_dir: str,
    model_dir: str,
    sd_model: str = "stabilityai/stable-diffusion-2-inpainting",
    resolution: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
) -> list[Path]:
    """Decode all JSON files in a directory to manga images.

    Returns list of output image paths.
    """
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(model_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load infiller
    controlnet = LayoutControlNet()
    controlnet_path = model_dir / "controlnet_best.pt"
    if controlnet_path.exists():
        controlnet.load_state_dict(torch.load(controlnet_path, map_location=device))
        logger.info("Loaded ControlNet from %s", controlnet_path)

    infiller = MangaInfiller(
        sd_model_id=sd_model,
        resolution=resolution,
        controlnet=controlnet,
    )
    infiller.load_sd_pipeline(device)

    # Auto-detect LoRA weights
    lora_path = model_dir / "unet_lora"
    if lora_path.exists() and (lora_path / "adapter_config.json").exists():
        infiller.load_lora_weights(lora_path)

    # Find JSON files
    json_paths = sorted(json_dir.rglob("*.json"))
    logger.info("Found %d JSON files in %s", len(json_paths), json_dir)

    output_paths = []
    for json_path in json_paths:
        try:
            page = MangaPage.from_json(json_path)
            image = decode_page(
                page, infiller,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )

            rel = json_path.relative_to(json_dir)
            img_path = output_dir / rel.with_suffix(".png")
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(str(img_path))
            output_paths.append(img_path)

            logger.info("Decoded %s → %s", json_path.name, img_path.name)
        except Exception as e:
            logger.error("Failed to decode %s: %s", json_path, e)

    return output_paths
