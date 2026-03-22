"""Artist index utilities for the 64K-artist pickle from manga-translate."""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def load_artist_index(pickle_path: str | Path) -> dict[str, list[str]]:
    """Load the artist→gallery-URLs mapping from a pickle file.

    The pickle is expected to be a dict mapping artist name (str) to a list of
    ExHentai gallery URLs (list[str]).
    """
    path = Path(pickle_path)
    if not path.exists():
        raise FileNotFoundError(f"Artist index pickle not found: {path}")

    with open(path, "rb") as f:
        index = pickle.load(f)

    if not isinstance(index, dict):
        raise TypeError(f"Expected dict, got {type(index).__name__}")

    logger.info("Loaded artist index: %d artists", len(index))
    return index


def select_top_artists(
    index: dict[str, list[str]],
    num_artists: int = 200,
    min_galleries: int = 5,
    max_galleries_per_artist: int = 20,
) -> dict[str, list[str]]:
    """Select top-N artists by gallery count, truncating each to max_galleries_per_artist.

    Artists with fewer than *min_galleries* galleries are excluded.
    Returns a new dict sorted by gallery count (descending).
    """
    # Filter by minimum
    eligible = {
        artist: urls
        for artist, urls in index.items()
        if len(urls) >= min_galleries
    }

    # Sort by gallery count descending, take top N
    ranked = sorted(eligible.items(), key=lambda kv: len(kv[1]), reverse=True)
    selected = ranked[:num_artists]

    # Truncate per-artist URLs
    result = {
        artist: urls[:max_galleries_per_artist]
        for artist, urls in selected
    }

    logger.info(
        "Selected %d artists (min_galleries=%d, max_per_artist=%d)",
        len(result),
        min_galleries,
        max_galleries_per_artist,
    )
    return result


def save_index_as_json(index: dict[str, list[str]], output_path: str | Path) -> Path:
    """Serialize artist index to JSON for upload to Modal volumes."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)
    logger.info("Saved artist index JSON: %s (%d artists)", path, len(index))
    return path


def load_index_from_json(json_path: str | Path) -> dict[str, list[str]]:
    """Load artist index from a JSON file (e.g. on a Modal volume)."""
    path = Path(json_path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)
