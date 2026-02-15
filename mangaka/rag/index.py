"""
Manga JSON embedding and vector index.

Builds a searchable index over a dataset of MangaPage JSONs.  Each page
is embedded along multiple axes so retrieval can match on narrative
content, visual composition, and element distribution simultaneously.

Embedding strategy (per page):
  - Description embedding:  sentence embedding of the page + panel descriptions
  - Structure embedding:    lightweight numerical features (panel count, element
                            type histogram, average bbox areas, reading-order
                            layout fingerprint)

The two are concatenated into a single vector per page, normalised, and
stored in a FAISS flat index for fast cosine-similarity search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from mangaka.schema import MangaPage, ELEMENT_TYPES, ELEMENT_TYPE_TO_ID

logger = logging.getLogger(__name__)

# Dimensionality of the structural feature vector
_NUM_STRUCT_FEATURES = (
    1                     # panel count (normalised)
    + len(ELEMENT_TYPES)  # element type histogram (6)
    + 1                   # total element count (normalised)
    + 1                   # avg panel area
    + 1                   # avg elements per panel
)  # = 10


class MangaIndex:
    """In-memory vector index over a manga JSON dataset.

    Supports building from a dataset directory (the output of
    ``build_dataset``) and querying with a free-text plot description.
    """

    def __init__(self):
        self.pages: list[MangaPage] = []
        self.paths: list[str] = []
        self.desc_embeds: np.ndarray | None = None   # (N, D_desc)
        self.struct_embeds: np.ndarray | None = None  # (N, _NUM_STRUCT_FEATURES)
        self.combined: np.ndarray | None = None       # (N, D_desc + _NUM_STRUCT_FEATURES)
        self._embed_model = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        dataset_dir: str | Path,
        embed_model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        """Build the index from a manga JSON dataset directory.

        *dataset_dir* is the output of ``scripts/build_dataset.py`` and
        should contain an ``index.json`` manifest plus a ``pages/``
        subdirectory of MangaPage JSON files.
        """
        dataset_dir = Path(dataset_dir)
        index_path = dataset_dir / "index.json"
        pages_dir = dataset_dir / "pages"

        if index_path.exists():
            manifest = json.loads(index_path.read_text())
            file_list = manifest.get("files", [])
        else:
            # Fall back to globbing
            file_list = [
                str(p.relative_to(pages_dir))
                for p in sorted(pages_dir.rglob("*.json"))
            ]

        logger.info("Loading %d pages from %s", len(file_list), dataset_dir)

        self.pages = []
        self.paths = []
        descriptions: list[str] = []

        for rel in file_list:
            json_path = pages_dir / rel
            if not json_path.exists():
                continue
            try:
                page = MangaPage.from_json(json_path)
                self.pages.append(page)
                self.paths.append(str(rel))
                descriptions.append(_page_to_description_text(page))
            except Exception as e:
                logger.warning("Skipping %s: %s", rel, e)

        if not self.pages:
            logger.error("No valid pages found in %s", dataset_dir)
            return

        logger.info("Loaded %d pages, computing embeddings...", len(self.pages))

        # Description embeddings
        self._embed_model = _load_embed_model(embed_model)
        self.desc_embeds = self._embed_model.encode(
            descriptions, batch_size=batch_size, show_progress_bar=True,
            normalize_embeddings=True,
        )

        # Structural embeddings
        struct = np.array(
            [_page_to_struct_features(p) for p in self.pages],
            dtype=np.float32,
        )
        # Normalise structural features to unit norm
        norms = np.linalg.norm(struct, axis=1, keepdims=True) + 1e-8
        self.struct_embeds = struct / norms

        # Combine (weight description more heavily than structure)
        desc_weight = 0.8
        struct_weight = 0.2
        self.combined = np.concatenate([
            self.desc_embeds * desc_weight,
            self.struct_embeds * struct_weight,
        ], axis=1)

        # Normalise combined vectors
        norms = np.linalg.norm(self.combined, axis=1, keepdims=True) + 1e-8
        self.combined = self.combined / norms

        logger.info(
            "Index built: %d pages, embed dim=%d",
            len(self.pages), self.combined.shape[1],
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "desc_embeds.npy", self.desc_embeds)
        np.save(path / "struct_embeds.npy", self.struct_embeds)
        np.save(path / "combined.npy", self.combined)

        meta = {
            "paths": self.paths,
            "pages_json": [p.model_dump(mode="json") for p in self.pages],
        }
        (path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8",
        )
        logger.info("Index saved to %s", path)

    def load(self, path: str | Path, embed_model: str = "all-MiniLM-L6-v2") -> None:
        """Load a previously saved index."""
        path = Path(path)
        self.desc_embeds = np.load(path / "desc_embeds.npy")
        self.struct_embeds = np.load(path / "struct_embeds.npy")
        self.combined = np.load(path / "combined.npy")

        meta = json.loads((path / "meta.json").read_text())
        self.paths = meta["paths"]
        self.pages = [MangaPage.model_validate(d) for d in meta["pages_json"]]
        self._embed_model = _load_embed_model(embed_model)

        logger.info("Index loaded: %d pages from %s", len(self.pages), path)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        plot: str,
        top_k: int = 5,
        structure_hint: dict | None = None,
    ) -> list[tuple[MangaPage, str, float]]:
        """Retrieve the most relevant manga pages for a plot description.

        Args:
            plot: free-text plot / story description from the user
            top_k: number of results to return
            structure_hint: optional dict to bias structural matching, e.g.
                ``{"panel_count": 5, "types": ["character", "speech_bubble"]}``

        Returns:
            List of (MangaPage, file_path, similarity_score) sorted by
            relevance (highest first).
        """
        if self.combined is None or self._embed_model is None:
            raise RuntimeError("Index not built or loaded.")

        # Embed the query
        q_desc = self._embed_model.encode(
            [plot], normalize_embeddings=True,
        )  # (1, D_desc)

        # Build query structural features from hint (or zeros)
        q_struct = np.zeros((1, _NUM_STRUCT_FEATURES), dtype=np.float32)
        if structure_hint:
            pc = structure_hint.get("panel_count", 0)
            q_struct[0, 0] = min(pc / 10.0, 1.0)
            for t in structure_hint.get("types", []):
                idx = ELEMENT_TYPE_TO_ID.get(t)
                if idx is not None:
                    q_struct[0, 1 + idx] = 1.0
            norm = np.linalg.norm(q_struct) + 1e-8
            q_struct = q_struct / norm

        q_combined = np.concatenate([q_desc * 0.8, q_struct * 0.2], axis=1)
        norm = np.linalg.norm(q_combined) + 1e-8
        q_combined = q_combined / norm

        # Cosine similarity
        scores = (self.combined @ q_combined.T).squeeze(1)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_indices:
            results.append((self.pages[i], self.paths[i], float(scores[i])))
        return results

    def query_by_page(
        self,
        reference: MangaPage,
        top_k: int = 5,
    ) -> list[tuple[MangaPage, str, float]]:
        """Find pages structurally and narratively similar to *reference*."""
        desc_text = _page_to_description_text(reference)
        struct = np.array(
            [_page_to_struct_features(reference)], dtype=np.float32,
        )
        norm = np.linalg.norm(struct) + 1e-8
        struct = struct / norm

        q_desc = self._embed_model.encode(
            [desc_text], normalize_embeddings=True,
        )
        q_combined = np.concatenate([q_desc * 0.8, struct * 0.2], axis=1)
        norm = np.linalg.norm(q_combined) + 1e-8
        q_combined = q_combined / norm

        scores = (self.combined @ q_combined.T).squeeze(1)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.pages[i], self.paths[i], float(scores[i])) for i in top_indices]


# ======================================================================
# Feature extraction helpers
# ======================================================================

def _page_to_description_text(page: MangaPage) -> str:
    """Concatenate all descriptions in a page into a single string for embedding."""
    parts = [page.description]
    for panel in page.panels:
        parts.append(panel.description)
        for elem in panel.elements:
            parts.append(f"{elem.type}: {elem.description}")
            if elem.text:
                parts.append(f'text: "{elem.text}"')
    return " | ".join(p for p in parts if p)


def _page_to_struct_features(page: MangaPage) -> list[float]:
    """Extract a numerical feature vector capturing page structure."""
    n_panels = len(page.panels)
    n_elements = sum(len(p.elements) for p in page.panels)

    # Element type histogram
    type_hist = [0.0] * len(ELEMENT_TYPES)
    for panel in page.panels:
        for elem in panel.elements:
            idx = ELEMENT_TYPE_TO_ID.get(elem.type)
            if idx is not None:
                type_hist[idx] += 1.0
    # Normalise histogram
    total = sum(type_hist) or 1.0
    type_hist = [v / total for v in type_hist]

    # Average panel area (normalised bbox)
    panel_areas = []
    for panel in page.panels:
        w = panel.bbox[2] - panel.bbox[0]
        h = panel.bbox[3] - panel.bbox[1]
        panel_areas.append(w * h)
    avg_area = sum(panel_areas) / len(panel_areas) if panel_areas else 0.0

    avg_elems_per_panel = n_elements / max(n_panels, 1)

    return [
        min(n_panels / 10.0, 1.0),
        *type_hist,
        min(n_elements / 50.0, 1.0),
        avg_area,
        min(avg_elems_per_panel / 10.0, 1.0),
    ]


def _load_embed_model(model_name: str):
    """Load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
