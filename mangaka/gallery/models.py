"""Pydantic data models for ExHentai gallery metadata."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

_GALLERY_URL_RE = re.compile(
    r"https?://(?:exhentai|e-hentai)\.org/g/(\d+)/([a-f0-9]+)"
)


def parse_gallery_url(url: str) -> tuple[int, str]:
    """Extract (gallery_id, token) from an ExHentai or E-Hentai gallery URL.

    Raises ``ValueError`` if *url* does not match the expected pattern.
    """
    m = _GALLERY_URL_RE.search(url)
    if not m:
        raise ValueError(f"Invalid gallery URL: {url}")
    return int(m.group(1)), m.group(2)


class GalleryMetadata(BaseModel):
    """Metadata scraped from an ExHentai gallery page."""

    gallery_id: int
    token: str
    title: str = ""
    artists: list[str] = Field(default_factory=list)
    language: str = ""
    page_count: int = 0
    url: str = ""
