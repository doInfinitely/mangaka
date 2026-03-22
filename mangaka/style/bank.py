"""Style bank — per-image embedding storage with top-k retrieval."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class StyleBank:
    """Stores one style embedding per image, grouped by artist.

    Storage layout on disk:
        bank_dir/
            artist_001.pt   # tensor of shape (N_images, style_dim)
            artist_002.pt
            ...

    In memory, embeddings are loaded lazily per artist and cached.
    """

    def __init__(self, bank_dir: str | Path, style_dim: int = 512):
        self.bank_dir = Path(bank_dir)
        self.style_dim = style_dim
        self._cache: dict[str, torch.Tensor] = {}

    @property
    def artist_ids(self) -> list[str]:
        """List available artists (from .pt files on disk)."""
        if not self.bank_dir.exists():
            return []
        return sorted(p.stem for p in self.bank_dir.glob("*.pt"))

    def save_artist(self, artist_id: str, embeddings: torch.Tensor):
        """Save embeddings for an artist.

        Args:
            artist_id: unique artist identifier
            embeddings: (N, style_dim) tensor of style embeddings
        """
        self.bank_dir.mkdir(parents=True, exist_ok=True)
        path = self.bank_dir / f"{artist_id}.pt"
        torch.save(embeddings.cpu(), path)
        self._cache.pop(artist_id, None)
        logger.info("Saved %d embeddings for artist '%s'", embeddings.shape[0], artist_id)

    def load_artist(self, artist_id: str, device: torch.device | None = None) -> torch.Tensor:
        """Load all embeddings for an artist.

        Args:
            artist_id: unique artist identifier
            device: target device (defaults to CPU)

        Returns:
            (N, style_dim) tensor of style embeddings
        """
        if artist_id not in self._cache:
            path = self.bank_dir / f"{artist_id}.pt"
            if not path.exists():
                raise FileNotFoundError(f"No style bank found for artist '{artist_id}'")
            self._cache[artist_id] = torch.load(path, map_location="cpu", weights_only=True)

        embeddings = self._cache[artist_id]
        if device is not None:
            embeddings = embeddings.to(device)
        return embeddings

    def top_k(
        self,
        query: torch.Tensor,
        artist_id: str,
        k: int = 256,
    ) -> torch.Tensor:
        """Retrieve the top-k most similar embeddings for a query.

        Args:
            query: (1, style_dim) or (style_dim,) query vector
            artist_id: artist whose bank to search
            k: number of nearest neighbors

        Returns:
            (k, style_dim) tensor of the k most similar embeddings
        """
        bank = self.load_artist(artist_id, device=query.device)
        n = bank.shape[0]
        k = min(k, n)

        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Cosine similarity
        query_norm = F.normalize(query, dim=-1)
        bank_norm = F.normalize(bank, dim=-1)
        sims = (query_norm @ bank_norm.T).squeeze(0)  # (N,)

        _, indices = sims.topk(k)
        return bank[indices]

    def get_full_bank(self, artist_id: str, device: torch.device | None = None) -> torch.Tensor:
        """Return the full unfiltered bank for an artist."""
        return self.load_artist(artist_id, device=device)

    def num_embeddings(self, artist_id: str) -> int:
        """Count how many embeddings an artist has."""
        bank = self.load_artist(artist_id)
        return bank.shape[0]

    def clear_cache(self):
        """Free cached embeddings from memory."""
        self._cache.clear()
