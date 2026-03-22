"""Disk-based gallery cache.

Layout::

    {cache_dir}/
        {gallery_id}/
            metadata.json
            pages/
                0000.png
                0001.png
                ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from PIL import Image

from .models import GalleryMetadata

logger = logging.getLogger(__name__)


class GalleryCache:
    """Persistent on-disk cache for downloaded gallery data."""

    def __init__(self, cache_dir: str | Path = "data/gallery_cache"):
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def _gallery_dir(self, gallery_id: int) -> Path:
        return self.root / str(gallery_id)

    def _meta_path(self, gallery_id: int) -> Path:
        return self._gallery_dir(gallery_id) / "metadata.json"

    def _pages_dir(self, gallery_id: int) -> Path:
        return self._gallery_dir(gallery_id) / "pages"

    # -- queries ----------------------------------------------------------------

    def has(self, gallery_id: int) -> bool:
        """Return True if metadata and at least one page image exist."""
        pages_dir = self._pages_dir(gallery_id)
        return (
            self._meta_path(gallery_id).exists()
            and pages_dir.exists()
            and any(pages_dir.iterdir())
        )

    def has_metadata(self, gallery_id: int) -> bool:
        return self._meta_path(gallery_id).exists()

    def get_metadata(self, gallery_id: int) -> GalleryMetadata:
        """Load cached metadata (fast, no images touched)."""
        text = self._meta_path(gallery_id).read_text(encoding="utf-8")
        return GalleryMetadata.model_validate_json(text)

    def get_images(self, gallery_id: int) -> list[Image.Image]:
        """Load all cached page images as PIL Images."""
        return [Image.open(p).convert("RGB") for p in self.get_image_paths(gallery_id)]

    def get_image_paths(self, gallery_id: int) -> list[Path]:
        """Return sorted list of cached page image paths."""
        pages_dir = self._pages_dir(gallery_id)
        if not pages_dir.exists():
            return []
        return sorted(
            p for p in pages_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )

    # -- writes -----------------------------------------------------------------

    def save_metadata(self, metadata: GalleryMetadata) -> Path:
        """Persist gallery metadata to disk."""
        gdir = self._gallery_dir(metadata.gallery_id)
        gdir.mkdir(parents=True, exist_ok=True)
        path = self._meta_path(metadata.gallery_id)
        path.write_text(
            metadata.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    def save_image(self, gallery_id: int, page_index: int, image: Image.Image) -> Path:
        """Save a single page image to the cache."""
        pages_dir = self._pages_dir(gallery_id)
        pages_dir.mkdir(parents=True, exist_ok=True)
        path = pages_dir / f"{page_index:04d}.png"
        image.save(path, format="PNG")
        return path
