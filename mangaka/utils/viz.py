"""
Visualization helpers for manga annotations and detections.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mangaka.schema import MangaPage, Panel, MangaElement, ELEMENT_TYPES
from mangaka.utils.bbox import denormalise_bbox


# Colour palette for element types
TYPE_COLOURS = {
    "character": (220, 50, 50),
    "speech_bubble": (50, 120, 220),
    "sfx": (220, 150, 30),
    "background": (100, 180, 80),
    "effect": (180, 60, 200),
    "object": (60, 200, 200),
}

PANEL_COLOUR = (255, 80, 80)


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a readable font, fall back to default."""
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_annotations(
    image: Image.Image,
    page: MangaPage,
    draw_panels: bool = True,
    draw_elements: bool = True,
    label_font_size: int = 14,
) -> Image.Image:
    """Draw bounding boxes and labels on a manga page image.

    Returns a copy of *image* with annotations overlaid.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(label_font_size)
    w, h = img.size

    for panel in page.panels:
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)

        if draw_panels:
            draw.rectangle([px1, py1, px2, py2], outline=PANEL_COLOUR, width=3)
            draw.text(
                (px1 + 4, max(0, py1 - label_font_size - 4)),
                f"panel",
                fill=PANEL_COLOUR,
                font=font,
            )

        if draw_elements:
            panel_w = px2 - px1
            panel_h = py2 - py1
            for elem in panel.elements:
                ex1, ey1, ex2, ey2 = denormalise_bbox(
                    elem.bbox, panel_w, panel_h,
                )
                # Element coords are relative to panel
                ex1 += px1
                ey1 += py1
                ex2 += px1
                ey2 += py1
                colour = TYPE_COLOURS.get(elem.type, (200, 200, 200))
                draw.rectangle([ex1, ey1, ex2, ey2], outline=colour, width=2)

                label = elem.type
                if elem.text:
                    label += f': "{elem.text[:20]}"'
                draw.text(
                    (ex1 + 2, max(0, ey1 - label_font_size - 2)),
                    label,
                    fill=colour,
                    font=font,
                )

    return img


def save_annotated(
    image: Image.Image,
    page: MangaPage,
    output_path: str | Path,
    **kwargs,
) -> None:
    """Draw annotations and save to *output_path*."""
    annotated = draw_annotations(image, page, **kwargs)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    annotated.save(str(output_path))


def side_by_side(
    original: Image.Image,
    generated: Image.Image,
    output_path: str | Path,
    label_left: str = "Original",
    label_right: str = "Generated",
) -> None:
    """Save two images side by side for comparison."""
    w = max(original.width, generated.width)
    h = max(original.height, generated.height)
    orig_resized = original.resize((w, h), Image.LANCZOS)
    gen_resized = generated.resize((w, h), Image.LANCZOS)

    combined = Image.new("RGB", (w * 2 + 20, h + 30), (40, 40, 40))
    combined.paste(orig_resized, (0, 30))
    combined.paste(gen_resized, (w + 20, 30))

    draw = ImageDraw.Draw(combined)
    font = _get_font(16)
    draw.text((10, 6), label_left, fill=(220, 220, 220), font=font)
    draw.text((w + 30, 6), label_right, fill=(220, 220, 220), font=font)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(str(output_path))
