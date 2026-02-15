"""
Layout renderer — converts MangaPage JSON into spatial conditioning maps
at each level of the containment hierarchy.

The infiller operates by descending the hierarchy:
  Page level:   layout shows panel borders + panel fills
  Panel level:  layout shows element bboxes + element type masks within a panel
  Element level: layout shows a single element fill (for fine detail rendering)

At each level the layout map is rendered at full SD resolution, giving
every container the full pixel budget — just like nissl's retina approach.

Channels (8 total):
  0: Container borders (panel borders at page level, element borders at panel level)
  1: Container fill (unique intensity per child)
  2-7: Element type masks (one channel per element type)
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from mangaka.schema import MangaPage, Panel, MangaElement, ELEMENT_TYPES, ELEMENT_TYPE_TO_ID
from mangaka.utils.bbox import denormalise_bbox


NUM_LAYOUT_CHANNELS = 2 + len(ELEMENT_TYPES)  # 8


# ---------------------------------------------------------------------------
# Page-level layout: shows panels
# ---------------------------------------------------------------------------

def render_page_layout(
    page: MangaPage,
    resolution: int = 512,
) -> np.ndarray:
    """Layout map for the full page — panels as children.

    Channel 0: panel borders
    Channel 1: panel fill (unique intensity per panel)
    Channels 2-7: element type masks (all elements at page scale)

    Returns:
        (NUM_LAYOUT_CHANNELS, resolution, resolution) float32 in [0, 1]
    """
    layout = np.zeros((NUM_LAYOUT_CHANNELS, resolution, resolution), dtype=np.float32)
    w, h = page.width, page.height
    sx = resolution / w if w > 0 else 1.0
    sy = resolution / h if h > 0 else 1.0

    border_img = Image.new("L", (resolution, resolution), 0)
    border_draw = ImageDraw.Draw(border_img)
    fill_img = Image.new("L", (resolution, resolution), 0)
    fill_draw = ImageDraw.Draw(fill_img)

    for pi, panel in enumerate(page.panels):
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
        lx1, ly1 = int(px1 * sx), int(py1 * sy)
        lx2, ly2 = int(px2 * sx), int(py2 * sy)

        border_draw.rectangle([lx1, ly1, lx2, ly2], outline=255, width=2)
        fill_val = int(255 * (pi + 1) / max(len(page.panels), 1))
        fill_draw.rectangle([lx1, ly1, lx2, ly2], fill=fill_val)

        # Also draw element type hints at page scale
        panel_w, panel_h = px2 - px1, py2 - py1
        if panel_w <= 0 or panel_h <= 0:
            continue
        for elem in panel.elements:
            ex1 = px1 + elem.bbox[0] * panel_w
            ey1 = py1 + elem.bbox[1] * panel_h
            ex2 = px1 + elem.bbox[2] * panel_w
            ey2 = py1 + elem.bbox[3] * panel_h
            _fill_type_channel(layout, elem.type, ex1, ey1, ex2, ey2, sx, sy, resolution)

    layout[0] = np.array(border_img, dtype=np.float32) / 255.0
    layout[1] = np.array(fill_img, dtype=np.float32) / 255.0
    return layout


# ---------------------------------------------------------------------------
# Panel-level layout: crop into a panel, shows its elements
# ---------------------------------------------------------------------------

def render_panel_layout(
    panel: Panel,
    resolution: int = 512,
) -> np.ndarray:
    """Layout map for a single panel — elements as children.

    The panel is treated as the full canvas (cropped from the page).
    Element bboxes are normalised relative to the panel, so they map
    directly to the [0, resolution] grid.

    Channel 0: element borders
    Channel 1: element fill (unique intensity per element)
    Channels 2-7: element type masks

    Returns:
        (NUM_LAYOUT_CHANNELS, resolution, resolution) float32 in [0, 1]
    """
    layout = np.zeros((NUM_LAYOUT_CHANNELS, resolution, resolution), dtype=np.float32)

    border_img = Image.new("L", (resolution, resolution), 0)
    border_draw = ImageDraw.Draw(border_img)
    fill_img = Image.new("L", (resolution, resolution), 0)
    fill_draw = ImageDraw.Draw(fill_img)

    for ei, elem in enumerate(panel.elements):
        # Element bbox is normalised to [0, 1] relative to the panel
        ex1 = int(elem.bbox[0] * resolution)
        ey1 = int(elem.bbox[1] * resolution)
        ex2 = int(elem.bbox[2] * resolution)
        ey2 = int(elem.bbox[3] * resolution)

        ex1 = max(0, min(ex1, resolution - 1))
        ey1 = max(0, min(ey1, resolution - 1))
        ex2 = max(ex1 + 1, min(ex2, resolution))
        ey2 = max(ey1 + 1, min(ey2, resolution))

        border_draw.rectangle([ex1, ey1, ex2, ey2], outline=255, width=2)
        fill_val = int(255 * (ei + 1) / max(len(panel.elements), 1))
        fill_draw.rectangle([ex1, ey1, ex2, ey2], fill=fill_val)

        _fill_type_channel(
            layout, elem.type,
            ex1, ey1, ex2, ey2,
            sx=1.0, sy=1.0, resolution=resolution,
            already_pixel=True,
        )

    layout[0] = np.array(border_img, dtype=np.float32) / 255.0
    layout[1] = np.array(fill_img, dtype=np.float32) / 255.0
    return layout


# ---------------------------------------------------------------------------
# Element-level layout: crop into an element, fill with type
# ---------------------------------------------------------------------------

def render_element_layout(
    element: MangaElement,
    resolution: int = 512,
) -> np.ndarray:
    """Layout map for a single element — the leaf level.

    The entire canvas represents the element crop. Channel 0 is a border
    around the whole region, channel 1 is a full fill, and the element's
    type channel is fully activated.

    Returns:
        (NUM_LAYOUT_CHANNELS, resolution, resolution) float32 in [0, 1]
    """
    layout = np.zeros((NUM_LAYOUT_CHANNELS, resolution, resolution), dtype=np.float32)

    # Full border and fill
    border_img = Image.new("L", (resolution, resolution), 0)
    ImageDraw.Draw(border_img).rectangle([0, 0, resolution - 1, resolution - 1], outline=255, width=2)
    layout[0] = np.array(border_img, dtype=np.float32) / 255.0
    layout[1] = 1.0  # full fill

    type_idx = ELEMENT_TYPE_TO_ID.get(element.type, 0)
    layout[2 + type_idx, :, :] = 1.0

    return layout


# ---------------------------------------------------------------------------
# Masks at each level
# ---------------------------------------------------------------------------

def render_panel_mask(
    panel: Panel,
    resolution: int = 512,
) -> np.ndarray:
    """Full-canvas mask for rendering an entire panel (used at page level).

    Returns (1, resolution, resolution) with 1s everywhere — the whole
    panel region is the inpainting target.
    """
    return np.ones((1, resolution, resolution), dtype=np.float32)


def render_element_mask_in_panel(
    panel: Panel,
    element_idx: int,
    resolution: int = 512,
) -> np.ndarray:
    """Mask for a single element within a panel-level crop.

    Returns (1, resolution, resolution) with 1s inside the element bbox.
    """
    mask = np.zeros((1, resolution, resolution), dtype=np.float32)
    elem = panel.elements[element_idx]
    ex1 = max(0, int(elem.bbox[0] * resolution))
    ey1 = max(0, int(elem.bbox[1] * resolution))
    ex2 = min(resolution, int(elem.bbox[2] * resolution))
    ey2 = min(resolution, int(elem.bbox[3] * resolution))
    mask[0, ey1:ey2, ex1:ex2] = 1.0
    return mask


def render_full_mask(resolution: int = 512) -> np.ndarray:
    """Full-canvas mask (for element-level leaf rendering).

    Returns (1, resolution, resolution) all 1s.
    """
    return np.ones((1, resolution, resolution), dtype=np.float32)


# ---------------------------------------------------------------------------
# Legacy / convenience
# ---------------------------------------------------------------------------

def render_layout_map(page: MangaPage, resolution: int = 512) -> np.ndarray:
    """Alias for render_page_layout (backward compat)."""
    return render_page_layout(page, resolution)


def render_inpaint_mask(
    page: MangaPage,
    panel_idx: int,
    element_idx: int | None = None,
    resolution: int = 512,
) -> np.ndarray:
    """Legacy mask renderer at page scale (backward compat)."""
    mask = np.zeros((1, resolution, resolution), dtype=np.float32)
    w, h = page.width, page.height
    sx = resolution / w if w > 0 else 1.0
    sy = resolution / h if h > 0 else 1.0

    panel = page.panels[panel_idx]
    px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)

    if element_idx is not None:
        elem = panel.elements[element_idx]
        panel_w, panel_h = px2 - px1, py2 - py1
        ex1 = px1 + elem.bbox[0] * panel_w
        ey1 = py1 + elem.bbox[1] * panel_h
        ex2 = px1 + elem.bbox[2] * panel_w
        ey2 = py1 + elem.bbox[3] * panel_h
    else:
        ex1, ey1, ex2, ey2 = px1, py1, px2, py2

    mx1, my1 = max(0, int(ex1 * sx)), max(0, int(ey1 * sy))
    mx2, my2 = min(resolution, int(ex2 * sx)), min(resolution, int(ey2 * sy))
    mask[0, my1:my2, mx1:mx2] = 1.0
    return mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_type_channel(
    layout: np.ndarray,
    elem_type: str,
    x1: float, y1: float, x2: float, y2: float,
    sx: float, sy: float,
    resolution: int,
    already_pixel: bool = False,
) -> None:
    """Fill the element type channel in the layout array."""
    if already_pixel:
        lx1, ly1, lx2, ly2 = int(x1), int(y1), int(x2), int(y2)
    else:
        lx1, ly1 = int(x1 * sx), int(y1 * sy)
        lx2, ly2 = int(x2 * sx), int(y2 * sy)

    lx1 = max(0, min(lx1, resolution - 1))
    ly1 = max(0, min(ly1, resolution - 1))
    lx2 = max(lx1 + 1, min(lx2, resolution))
    ly2 = max(ly1 + 1, min(ly2, resolution))

    type_idx = ELEMENT_TYPE_TO_ID.get(elem_type, 0)
    layout[2 + type_idx, ly1:ly2, lx1:lx2] = 1.0


def layout_to_pil(layout: np.ndarray) -> Image.Image:
    """Convert layout map to a visualisable RGB image for debugging."""
    h, w = layout.shape[1], layout.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    # Borders in white
    rgb[:, :, 0] = (layout[0] * 200).astype(np.uint8)
    rgb[:, :, 1] = (layout[0] * 200).astype(np.uint8)
    rgb[:, :, 2] = (layout[0] * 200).astype(np.uint8)

    colours = [
        (220, 50, 50),    # character
        (50, 120, 220),   # speech_bubble
        (220, 150, 30),   # sfx
        (100, 180, 80),   # background
        (180, 60, 200),   # effect
        (60, 200, 200),   # object
    ]

    for i, colour in enumerate(colours):
        channel = layout[2 + i]
        for c in range(3):
            rgb[:, :, c] = np.maximum(
                rgb[:, :, c],
                (channel * colour[c]).astype(np.uint8),
            )

    return Image.fromarray(rgb)
