"""High-level gallery fetching and artist image collection."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from tqdm import tqdm

from .cache import GalleryCache
from .client import ExHentaiClient
from .models import GalleryMetadata, parse_gallery_url

logger = logging.getLogger(__name__)


def fetch_gallery(
    url: str,
    client: ExHentaiClient,
    cache: GalleryCache,
) -> GalleryMetadata:
    """Fetch and cache a single gallery. Returns metadata (fast on cache hit)."""
    gallery_id, _ = parse_gallery_url(url)

    if cache.has(gallery_id):
        logger.info("Cache hit for gallery %d", gallery_id)
        return cache.get_metadata(gallery_id)

    # Fetch metadata
    if cache.has_metadata(gallery_id):
        metadata = cache.get_metadata(gallery_id)
    else:
        metadata = client.fetch_metadata(url)
        cache.save_metadata(metadata)

    # Download pages
    page_urls = client.get_page_urls(url)
    logger.info("Downloading %d pages for gallery %d (%s)", len(page_urls), gallery_id, metadata.title)

    for idx, page_url in enumerate(tqdm(page_urls, desc=f"Gallery {gallery_id}", leave=False)):
        try:
            image = client.download_image(page_url)
            cache.save_image(gallery_id, idx, image)
        except Exception as exc:
            logger.warning("Failed to download page %d of gallery %d: %s", idx, gallery_id, exc)

    return metadata


def passes_artist_filter(metadata: GalleryMetadata, allowed_artists: set[str]) -> bool:
    """Return True only if ALL artists in the gallery are in the allowed set.

    This prevents style leakage from collaborating artists outside the selection.
    Galleries with no tagged artists are skipped.
    """
    if not metadata.artists:
        return False
    normalised_allowed = {a.lower().strip() for a in allowed_artists}
    return all(a.lower().strip() in normalised_allowed for a in metadata.artists)


def fetch_artist_images(
    artist_names: list[str],
    client: ExHentaiClient,
    cache: GalleryCache,
    output_dir: str | Path,
    max_galleries_per_artist: int = 20,
) -> dict[str, list[Path]]:
    """Search, download, filter, and organise images by artist.

    Output layout matches ``StyleTransferDataset`` expectation::

        {output_dir}/
            artist_a/
                0000.png
                0001.png
            artist_b/
                ...

    Returns a mapping from artist name to list of image paths.
    """
    output_dir = Path(output_dir)
    allowed = set(artist_names)
    result: dict[str, list[Path]] = {}

    for artist in artist_names:
        logger.info("Searching galleries for artist: %s", artist)
        artist_dir = output_dir / artist.replace(" ", "_")
        artist_dir.mkdir(parents=True, exist_ok=True)

        gallery_urls = client.search_artist_galleries(artist, max_results=max_galleries_per_artist)
        logger.info("Found %d galleries for '%s'", len(gallery_urls), artist)

        image_paths: list[Path] = []
        img_counter = 0

        for gallery_url in tqdm(gallery_urls, desc=f"Artist: {artist}"):
            gallery_id, _ = parse_gallery_url(gallery_url)

            # Fast metadata check for artist filtering
            if cache.has_metadata(gallery_id):
                meta = cache.get_metadata(gallery_id)
            else:
                try:
                    meta = client.fetch_metadata(gallery_url)
                    cache.save_metadata(meta)
                except Exception as exc:
                    logger.warning("Failed to fetch metadata for %s: %s", gallery_url, exc)
                    continue

            if not passes_artist_filter(meta, allowed):
                skipped_artists = [a for a in meta.artists if a.lower().strip() not in {x.lower().strip() for x in allowed}]
                logger.info(
                    "Skipping gallery %d — has artists outside selection: %s",
                    gallery_id,
                    skipped_artists,
                )
                continue

            # Ensure full gallery is cached
            try:
                fetch_gallery(gallery_url, client, cache)
            except Exception as exc:
                logger.warning("Failed to fetch gallery %d: %s", gallery_id, exc)
                continue

            # Copy images to output dir
            for src_path in cache.get_image_paths(gallery_id):
                dst = artist_dir / f"{img_counter:04d}.png"
                shutil.copy2(src_path, dst)
                image_paths.append(dst)
                img_counter += 1

        result[artist] = image_paths
        logger.info("Collected %d images for artist '%s'", len(image_paths), artist)

    return result
