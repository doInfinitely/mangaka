"""
Encode pipeline: manga image → MangaPage JSON using the trained detector.

Runs hierarchical detection:
  1. Detect panels on the full page
  2. For each panel, crop and detect elements
  3. Generate descriptions via GPT-2 head
  4. Assemble into MangaPage JSON
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import GPT2Tokenizer

from mangaka.schema import MangaPage, Panel, MangaElement, ID_TO_ELEMENT_TYPE
from mangaka.detector.model import (
    MangaDetectorNet,
    TYPE_NONE,
    TYPE_PANEL,
    TYPE_OFFSET,
    RETINA_SIZE,
    NUM_TYPE_CLASSES,
)
from mangaka.utils.bbox import scale_and_pad, retina_to_source, normalise_bbox

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


@torch.no_grad()
def encode_page(
    image: Image.Image,
    model: MangaDetectorNet,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    max_detections: int = 50,
) -> MangaPage:
    """Encode a single manga page image into a MangaPage JSON.

    Args:
        image: PIL Image of the manga page
        model: trained MangaDetectorNet
        tokenizer: GPT-2 tokenizer for description generation
        device: torch device
        max_detections: max detections per level

    Returns:
        MangaPage with detected panels and elements
    """
    model.eval()
    w, h = image.size

    # Level 1: detect panels on full page
    panels = _detect_level(
        image, model, tokenizer, device,
        expected_type=TYPE_PANEL,
        max_detections=max_detections,
    )

    manga_panels = []
    for panel_bbox_retina, panel_desc, _ in panels:
        # Convert panel bbox to normalised page coordinates
        retina_img, scale, ox, oy = scale_and_pad(image)
        panel_bbox_source = retina_to_source(panel_bbox_retina, scale, ox, oy)

        # Clamp to image bounds
        px1 = max(0, min(panel_bbox_source[0], w))
        py1 = max(0, min(panel_bbox_source[1], h))
        px2 = max(0, min(panel_bbox_source[2], w))
        py2 = max(0, min(panel_bbox_source[3], h))

        if px2 <= px1 or py2 <= py1:
            continue

        panel_bbox_norm = normalise_bbox((px1, py1, px2, py2), w, h)

        # Level 2: detect elements within panel crop
        panel_crop = image.crop((px1, py1, px2, py2))
        pw, ph = panel_crop.size

        elements = _detect_level(
            panel_crop, model, tokenizer, device,
            expected_type=None,  # any element type
            max_detections=max_detections,
        )

        manga_elements = []
        for elem_bbox_retina, elem_desc, elem_type_id in elements:
            retina_crop, cs, cox, coy = scale_and_pad(panel_crop)
            elem_bbox_source = retina_to_source(elem_bbox_retina, cs, cox, coy)

            # Clamp to panel bounds
            ex1 = max(0, min(elem_bbox_source[0], pw))
            ey1 = max(0, min(elem_bbox_source[1], ph))
            ex2 = max(0, min(elem_bbox_source[2], pw))
            ey2 = max(0, min(elem_bbox_source[3], ph))

            if ex2 <= ex1 or ey2 <= ey1:
                continue

            elem_bbox_norm = normalise_bbox((ex1, ey1, ex2, ey2), pw, ph)
            elem_type_idx = elem_type_id - TYPE_OFFSET
            elem_type = ID_TO_ELEMENT_TYPE.get(elem_type_idx, "object")

            # Extract text for speech bubbles / SFX from description
            text = None
            if elem_type in ("speech_bubble", "sfx") and '"' in elem_desc:
                # Try to extract quoted text from description
                import re
                match = re.search(r'"([^"]+)"', elem_desc)
                if match:
                    text = match.group(1)

            manga_elements.append(MangaElement(
                bbox=elem_bbox_norm,
                type=elem_type,
                description=elem_desc,
                text=text,
            ))

        manga_panels.append(Panel(
            bbox=panel_bbox_norm,
            description=panel_desc,
            elements=manga_elements,
        ))

    # Page-level description: concatenate panel descriptions
    page_desc = "; ".join(p.description for p in manga_panels) if manga_panels else ""

    return MangaPage(
        width=w,
        height=h,
        description=page_desc,
        panels=manga_panels,
    )


def _detect_level(
    image: Image.Image,
    model: MangaDetectorNet,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    expected_type: int | None = None,
    max_detections: int = 50,
) -> list[tuple[tuple[float, ...], str, int]]:
    """Run autoregressive detection at one level.

    Returns list of (retina_bbox, description, type_id).
    """
    retina_img, scale, ox, oy = scale_and_pad(image)
    img_t = (
        torch.from_numpy(np.array(retina_img))
        .permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    detections = []
    prev_bbox = torch.zeros(1, 5, device=device)  # NONE

    for _ in range(max_detections):
        bbox_pred, type_pred, desc_texts = model(
            img_t, prev_bbox,
            generate=True,
            tokenizer=tokenizer,
        )

        type_id = type_pred.argmax(dim=1).item()

        # Stop on NONE
        if type_id == TYPE_NONE:
            break

        # If we expect a specific type (e.g., panels), skip others
        if expected_type is not None and type_id != expected_type:
            break

        # Get predicted bbox in retina coords
        rb = (bbox_pred[0] * RETINA_SIZE).tolist()
        desc = desc_texts[0] if desc_texts else ""

        detections.append((tuple(rb), desc, type_id))

        # Update prev_bbox for next iteration
        prev_bbox = torch.tensor(
            [[rb[0], rb[1], rb[2], rb[3], float(type_id)]],
            device=device,
        )

    return detections


def encode_directory(
    image_dir: str,
    output_dir: str,
    model_path: str,
    backbone: str = "resnet50",
    gpt2_model: str = "gpt2",
    max_detections: int = 50,
) -> list[Path]:
    """Encode all manga images in a directory to JSON.

    Returns list of output JSON paths.
    """
    from safetensors.torch import load_file as safetensors_load

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MangaDetectorNet(backbone=backbone, gpt2_model=gpt2_model).to(device)
    state = safetensors_load(model_path, device=str(device))
    model.load_state_dict(state)
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)

    # Find images
    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )
    logger.info("Found %d images in %s", len(image_paths), image_dir)

    output_paths = []
    for img_path in image_paths:
        try:
            image = Image.open(img_path).convert("RGB")
            page = encode_page(image, model, tokenizer, device, max_detections)

            rel = img_path.relative_to(image_dir)
            json_path = output_dir / rel.with_suffix(".json")
            json_path.parent.mkdir(parents=True, exist_ok=True)
            page.to_json(json_path)
            output_paths.append(json_path)

            logger.info(
                "Encoded %s → %d panels, %d elements",
                img_path.name, page.num_panels, page.num_elements,
            )
        except Exception as e:
            logger.error("Failed to encode %s: %s", img_path, e)

    return output_paths
