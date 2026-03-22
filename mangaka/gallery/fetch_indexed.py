"""Fetch artist images using pre-supplied gallery URLs from the index.

Unlike ``fetch_artist_images`` (which searches ExHentai), this takes gallery
URLs directly from the artist index pickle — skipping the search step entirely.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from .cache import GalleryCache
from .client import ExHentaiClient
from .fetch import fetch_gallery
from .models import parse_gallery_url

logger = logging.getLogger(__name__)


def fetch_artist_from_index(
    artist_name: str,
    gallery_urls: list[str],
    client: ExHentaiClient,
    cache: GalleryCache,
    output_dir: str | Path,
) -> list[Path]:
    """Fetch images for a single artist given pre-supplied gallery URLs.

    Downloads each gallery (using cache), copies images to::

        {output_dir}/{artist_id}/0000.png, 0001.png, ...

    Individual gallery failures are logged and skipped — the function always
    returns whatever images were successfully collected.
    """
    output_dir = Path(output_dir)
    artist_id = artist_name.replace(" ", "_")
    artist_dir = output_dir / artist_id
    artist_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    img_counter = 0

    for gallery_url in gallery_urls:
        try:
            gallery_id, _ = parse_gallery_url(gallery_url)
        except ValueError as exc:
            logger.warning("Invalid gallery URL %s: %s", gallery_url, exc)
            continue

        try:
            fetch_gallery(gallery_url, client, cache)
        except Exception as exc:
            logger.warning(
                "Failed to fetch gallery %d for artist '%s': %s",
                gallery_id,
                artist_name,
                exc,
            )
            continue

        for src_path in cache.get_image_paths(gallery_id):
            dst = artist_dir / f"{img_counter:04d}.png"
            shutil.copy2(src_path, dst)
            image_paths.append(dst)
            img_counter += 1

    logger.info(
        "Fetched %d images for artist '%s' from %d galleries",
        len(image_paths),
        artist_name,
        len(gallery_urls),
    )
    return image_paths
