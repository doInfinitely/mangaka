"""Parallel style preparation — fetch galleries + encode embeddings for artists.

Called during the "analyzing" phase so style banks are ready by decode time.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def prepare_artist_styles(
    artist_names: list[str],
    cache_dir: str | Path = "data/gallery_cache",
    image_dir: str | Path = "data/images",
    bank_dir: str | Path = "data/style_bank",
    style_dim: int = 512,
    device: str = "auto",
    resolution: int = 512,
    batch_size: int = 64,
    max_galleries_per_artist: int = 20,
) -> dict[str, dict[str, int]]:
    """Fetch galleries and encode style embeddings for a list of artists.

    IO-bound gallery fetching runs in parallel (one thread per artist).
    GPU encoding runs sequentially per artist after all images are fetched.

    Args:
        artist_names: Artist names to prepare.
        cache_dir: Gallery cache directory.
        image_dir: Output directory for downloaded images.
        bank_dir: Directory for style bank .pt files.
        style_dim: Style embedding dimension.
        device: Torch device ("auto", "cuda", "cpu", "mps").
        resolution: Image resolution for encoding.
        batch_size: Batch size for GPU encoding.
        max_galleries_per_artist: Max galleries to fetch per artist.

    Returns:
        Status dict: ``{ "artist_name": { "images": N, "embeddings": N } }``
    """
    from mangaka.style.bank import StyleBank

    bank_dir = Path(bank_dir)
    image_dir = Path(image_dir)
    bank = StyleBank(bank_dir, style_dim=style_dim)

    # Skip artists whose bank already exists (cache hit)
    to_fetch: list[str] = []
    status: dict[str, dict[str, int]] = {}

    for name in artist_names:
        artist_id = name.replace(" ", "_")
        bank_path = bank_dir / f"{artist_id}.pt"
        if bank_path.exists():
            try:
                n = bank.num_embeddings(artist_id)
                status[name] = {"images": n, "embeddings": n}
                logger.info("Style bank cache hit for '%s' (%d embeddings)", name, n)
            except Exception:
                to_fetch.append(name)
        else:
            to_fetch.append(name)

    if not to_fetch:
        return status

    # --- Stage 1: Parallel gallery fetching (IO-bound) -----------------------
    fetched_images = _fetch_galleries_parallel(
        to_fetch, str(cache_dir), str(image_dir), max_galleries_per_artist
    )

    # --- Stage 2: Sequential GPU encoding ------------------------------------
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
        else:
            dev = torch.device("cpu")
    else:
        dev = torch.device(device)

    from mangaka.style.encoder import StyleEncoder

    encoder = StyleEncoder(style_dim=style_dim).to(dev)
    encoder.eval()

    transform = transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
    ])

    for name in to_fetch:
        artist_id = name.replace(" ", "_")
        paths = fetched_images.get(name, [])

        if not paths:
            logger.warning("No images found for artist '%s'", name)
            status[name] = {"images": 0, "embeddings": 0}
            continue

        all_embeddings = _encode_images(paths, encoder, transform, dev, batch_size, resolution)
        bank.save_artist(artist_id, all_embeddings)
        status[name] = {"images": len(paths), "embeddings": all_embeddings.shape[0]}
        logger.info("Encoded %d embeddings for '%s'", all_embeddings.shape[0], name)

    return status


def _fetch_galleries_parallel(
    artist_names: list[str],
    cache_dir: str,
    image_dir: str,
    max_galleries: int,
) -> dict[str, list[Path]]:
    """Fetch gallery images for multiple artists in parallel threads."""
    from mangaka.gallery import ExHentaiClient, GalleryCache, fetch_artist_images

    def _fetch_single(artist: str) -> tuple[str, list[Path]]:
        try:
            client = ExHentaiClient()
            cache = GalleryCache(cache_dir)
            result = fetch_artist_images(
                [artist], client, cache, image_dir,
                max_galleries_per_artist=max_galleries,
            )
            return artist, result.get(artist, [])
        except Exception as exc:
            logger.warning("Failed to fetch galleries for '%s': %s", artist, exc)
            return artist, []

    results: dict[str, list[Path]] = {}
    with ThreadPoolExecutor(max_workers=len(artist_names)) as pool:
        futures = {pool.submit(_fetch_single, name): name for name in artist_names}
        for future in futures:
            name, paths = future.result()
            results[name] = paths

    return results


def _encode_images(
    paths: list[Path],
    encoder: torch.nn.Module,
    transform: transforms.Compose,
    device: torch.device,
    batch_size: int,
    resolution: int,
) -> torch.Tensor:
    """Encode a list of image paths into style embeddings."""
    from PIL import Image

    all_embeddings: list[torch.Tensor] = []

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch_tensors = []
        for img_path in batch_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                batch_tensors.append(transform(img))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", img_path, exc)
                batch_tensors.append(torch.zeros(3, resolution, resolution))

        batch = torch.stack(batch_tensors).to(device)
        embeddings = encoder.encode_batch(batch)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)
