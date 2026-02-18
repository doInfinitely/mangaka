"""
Core JSON schema for manga page representations.

Every manga page is encoded as a hierarchical structure:
  MangaPage -> Panel -> MangaElement

Bounding boxes are normalised to [0, 1] relative to the parent container
(page for panels, panel for elements).  This makes the representation
resolution-independent and simplifies both training and generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Element types
# ---------------------------------------------------------------------------
ELEMENT_TYPES = (
    "character",
    "speech_bubble",
    "sfx",
    "background",
    "effect",
    "object",
)

ElementType = Literal[
    "character",
    "speech_bubble",
    "sfx",
    "background",
    "effect",
    "object",
]

NUM_ELEMENT_TYPES = len(ELEMENT_TYPES)
ELEMENT_TYPE_TO_ID = {t: i for i, t in enumerate(ELEMENT_TYPES)}
ID_TO_ELEMENT_TYPE = {i: t for i, t in enumerate(ELEMENT_TYPES)}


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------
class MangaElement(BaseModel):
    """A single visual element within a manga panel."""

    bbox: tuple[float, float, float, float] = Field(
        ...,
        description=(
            "Bounding box as (x1, y1, x2, y2) normalised to [0, 1] "
            "relative to the parent panel."
        ),
    )
    type: ElementType = Field(
        ...,
        description="Semantic type of this element.",
    )
    description: str = Field(
        ...,
        description="Free-form natural-language description of the element.",
    )
    text: str | None = Field(
        default=None,
        description="Text content (for speech_bubble and sfx only).",
    )


class Panel(BaseModel):
    """A single panel (frame) within a manga page."""

    bbox: tuple[float, float, float, float] = Field(
        ...,
        description=(
            "Bounding box as (x1, y1, x2, y2) normalised to [0, 1] "
            "relative to the page."
        ),
    )
    description: str = Field(
        ...,
        description="Overall description of the panel composition and mood.",
    )
    elements: list[MangaElement] = Field(
        default_factory=list,
        description="Visual elements contained in this panel.",
    )


class MangaPage(BaseModel):
    """Complete structured representation of a single manga page."""

    width: int = Field(..., description="Original image width in pixels.")
    height: int = Field(..., description="Original image height in pixels.")
    description: str = Field(
        ...,
        description="Page-level narrative summary.",
    )
    style: str | None = Field(
        default=None,
        description="Art style for rendering (e.g. 'manga, black and white, ink drawing').",
    )
    panels: list[Panel] = Field(
        default_factory=list,
        description="Panels on this page, in reading order.",
    )

    # -- Serialisation helpers -----------------------------------------------

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialise to JSON string, optionally writing to *path*."""
        data = self.model_dump(mode="json")
        text = json.dumps(data, indent=indent, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, path_or_str: str | Path) -> "MangaPage":
        """Load from a JSON file path or raw JSON string."""
        p = Path(path_or_str)
        if p.exists():
            text = p.read_text(encoding="utf-8")
        else:
            text = str(path_or_str)
        return cls.model_validate_json(text)

    # -- Convenience ---------------------------------------------------------

    @property
    def num_panels(self) -> int:
        return len(self.panels)

    @property
    def num_elements(self) -> int:
        return sum(len(p.elements) for p in self.panels)
