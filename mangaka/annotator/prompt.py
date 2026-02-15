"""
Shared prompt templates and response parsers for manga annotation.

Two-pass strategy:
  1. Detection pass: identify panels and elements with bounding boxes
  2. Description pass: describe each element in detail from a crop
"""

from __future__ import annotations

import json
import logging
import re

from mangaka.schema import MangaPage, Panel, MangaElement

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DETECTION_SYSTEM = """\
You are a manga analysis expert. You identify panels and visual elements in \
manga pages with precise bounding boxes. You always respond with valid JSON."""

DESCRIPTION_SYSTEM = """\
You are a manga art expert. You provide detailed, vivid descriptions of manga \
visual elements. Respond with a single JSON object."""


# ---------------------------------------------------------------------------
# Detection pass
# ---------------------------------------------------------------------------

DETECTION_PROMPT = """\
Analyze this manga page image. Identify every panel and every visual element \
within each panel.

For each panel, provide:
- bbox: [x1, y1, x2, y2] as fractions of the full page (0.0 to 1.0)
- description: brief description of the panel
- elements: list of elements within the panel

For each element, provide:
- bbox: [x1, y1, x2, y2] as fractions of the PANEL (not the page) (0.0 to 1.0)
- type: one of "character", "speech_bubble", "sfx", "background", "effect", "object"
- description: brief description
- text: the text content if this is a speech_bubble or sfx, otherwise null

Return ONLY a JSON object with this exact structure:
{
  "description": "page-level narrative summary",
  "panels": [
    {
      "bbox": [x1, y1, x2, y2],
      "description": "panel description",
      "elements": [
        {
          "bbox": [x1, y1, x2, y2],
          "type": "character",
          "description": "element description",
          "text": null
        }
      ]
    }
  ]
}

Be thorough — identify ALL panels and ALL visible elements. Bounding boxes \
must be tight and accurate."""


# ---------------------------------------------------------------------------
# Description pass (per-element crop)
# ---------------------------------------------------------------------------

DESCRIPTION_PROMPT = """\
This is a cropped element from a manga panel. The element type is: {element_type}.

Provide a detailed visual description of this element. Include:
- Visual appearance (pose, expression, clothing, style)
- Art style details (line weight, shading, hatching)
- Spatial relationships and composition
- Any text visible

Return ONLY a JSON object:
{{"description": "your detailed description here"}}"""


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

def _extract_json(raw: str) -> dict:
    """Extract JSON from a response that may contain markdown fences."""
    # Try direct parse first
    raw = raw.strip()
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: find first { ... } block
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = None

    raise ValueError(f"Could not extract JSON from response: {raw[:200]}...")


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(v, hi))


def _parse_bbox(raw: list) -> tuple[float, float, float, float]:
    """Parse and clamp a bbox list to a normalised tuple."""
    if len(raw) != 4:
        raise ValueError(f"Expected 4-element bbox, got {len(raw)}")
    return (
        _clamp(float(raw[0])),
        _clamp(float(raw[1])),
        _clamp(float(raw[2])),
        _clamp(float(raw[3])),
    )


def parse_detection_response(raw: str, width: int, height: int) -> MangaPage:
    """Parse the detection pass JSON into a MangaPage."""
    data = _extract_json(raw)

    panels = []
    for p in data.get("panels", []):
        elements = []
        for e in p.get("elements", []):
            elem_type = e.get("type", "object")
            if elem_type not in (
                "character", "speech_bubble", "sfx",
                "background", "effect", "object",
            ):
                elem_type = "object"
            elements.append(MangaElement(
                bbox=_parse_bbox(e["bbox"]),
                type=elem_type,
                description=e.get("description", ""),
                text=e.get("text"),
            ))
        panels.append(Panel(
            bbox=_parse_bbox(p["bbox"]),
            description=p.get("description", ""),
            elements=elements,
        ))

    return MangaPage(
        width=width,
        height=height,
        description=data.get("description", ""),
        panels=panels,
    )


def parse_description_response(raw: str) -> str | None:
    """Parse the description pass JSON, returning the description string."""
    try:
        data = _extract_json(raw)
        return data.get("description")
    except (ValueError, KeyError) as e:
        logger.warning("Failed to parse description response: %s", e)
        return None
