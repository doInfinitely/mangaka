"""
Dataset for training MangaDetectorNet.

Loads annotation JSONs (MangaPage) + source images and produces teacher-forced
samples for the two-level hierarchy:

  Level 1 — Panels:  (page_retina, prev_panel_bbox, target_panel_bbox + type)
  Level 2 — Elements: (panel_retina, prev_elem_bbox, target_elem_bbox + type)

Each sample also carries tokenised description text for the GPT-2 head.

Data format on disk:
    image_dir/
        chapter01/page001.jpg
        chapter01/page002.jpg
        ...
    annotation_dir/
        chapter01/page001.json   # MangaPage JSON
        chapter01/page002.json
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from mangaka.schema import (
    MangaPage,
    ELEMENT_TYPE_TO_ID,
)
from mangaka.detector.model import TYPE_NONE, TYPE_PANEL, TYPE_OFFSET, RETINA_SIZE
from mangaka.utils.bbox import scale_and_pad, denormalise_bbox

logger = logging.getLogger(__name__)

PREV_BBOX_NONE = (0.0, 0.0, 0.0, 0.0, float(TYPE_NONE))


class MangaDetectorDataset(Dataset):
    """Teacher-forced dataset for hierarchical manga detection + description.

    Each sample yields:
        img_tensor:      (3, 1024, 1024) float [0, 1]
        prev_bbox:       (5,) — (x1, y1, x2, y2, type_id) in retina coords
        target_bbox:     (5,) — target (x1, y1, x2, y2, type_id) in retina coords
        desc_input_ids:  (max_desc_len,) — tokenised description (padded)
        desc_attn_mask:  (max_desc_len,) — attention mask
        target_retina_bbox: (4,) — target bbox in retina coords (for ROI pooling)
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_dir: str | Path,
        tokenizer: GPT2Tokenizer | None = None,
        max_desc_len: int = 64,
        retina_size: int = RETINA_SIZE,
    ):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.retina_size = retina_size
        self.max_desc_len = max_desc_len

        # Tokeniser for description text
        if tokenizer is None:
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer

        # Discover and index all annotation files
        self.pages: list[tuple[Path, Path, MangaPage]] = []
        self.index: list[tuple[int, str, tuple, int]] = []
        self._discover_and_index()

    def _discover_and_index(self):
        """Find all annotation JSONs and build the teacher-forced sample index."""
        json_paths = sorted(self.annotation_dir.rglob("*.json"))
        logger.info("Found %d annotation files in %s", len(json_paths), self.annotation_dir)

        for json_path in json_paths:
            rel = json_path.relative_to(self.annotation_dir)
            # Try common image extensions
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                candidate = self.image_dir / rel.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                logger.warning("No image found for %s, skipping", json_path)
                continue

            try:
                page = MangaPage.from_json(json_path)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", json_path, e)
                continue

            page_idx = len(self.pages)
            self.pages.append((img_path, json_path, page))

            # Level 1: panel detection samples
            n_panels = len(page.panels)
            for i in range(n_panels + 1):
                self.index.append((page_idx, "panel", (), i))

            # Level 2: element detection samples (within each panel)
            for pi, panel in enumerate(page.panels):
                n_elems = len(panel.elements)
                for i in range(n_elems + 1):
                    self.index.append((page_idx, "element", (pi,), i))

        logger.info(
            "Indexed %d pages, %d samples",
            len(self.pages), len(self.index),
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        page_idx, level, parent_ids, item_idx = self.index[idx]
        img_path, _, page = self.pages[page_idx]

        if level == "panel":
            return self._make_panel_sample(img_path, page, item_idx)
        else:
            return self._make_element_sample(img_path, page, parent_ids[0], item_idx)

    # -- Panel-level samples ------------------------------------------------

    def _make_panel_sample(self, img_path: Path, page: MangaPage, item_idx: int):
        """Generate a panel-level detection sample."""
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        retina, scale, ox, oy = scale_and_pad(image, retina_size=self.retina_size)

        panels = page.panels

        # Previous panel bbox
        if item_idx > 0 and item_idx - 1 < len(panels):
            prev_panel = panels[item_idx - 1]
            pb = self._norm_to_retina(prev_panel.bbox, w, h, scale, ox, oy)
            prev = (*pb, float(TYPE_PANEL))
        else:
            prev = PREV_BBOX_NONE

        # Target panel bbox
        if item_idx < len(panels):
            target_panel = panels[item_idx]
            tb = self._norm_to_retina(target_panel.bbox, w, h, scale, ox, oy)
            target = (*tb, float(TYPE_PANEL))
            description = target_panel.description
            target_retina = tb
        else:
            target = (0.0, 0.0, 0.0, 0.0, float(TYPE_NONE))
            description = ""
            target_retina = (0.0, 0.0, 0.0, 0.0)

        return self._pack_sample(retina, prev, target, target_retina, description)

    # -- Element-level samples ----------------------------------------------

    def _make_element_sample(
        self, img_path: Path, page: MangaPage, panel_idx: int, item_idx: int,
    ):
        """Generate an element-level detection sample (within a panel crop)."""
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        panel = page.panels[panel_idx]

        # Crop the panel region
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
        panel_crop = image.crop((px1, py1, px2, py2))
        pw, ph = panel_crop.size

        retina, scale, ox, oy = scale_and_pad(panel_crop, retina_size=self.retina_size)
        elements = panel.elements

        # Previous element bbox
        if item_idx > 0 and item_idx - 1 < len(elements):
            prev_elem = elements[item_idx - 1]
            pb = self._norm_to_retina(prev_elem.bbox, pw, ph, scale, ox, oy)
            type_id = TYPE_OFFSET + ELEMENT_TYPE_TO_ID[prev_elem.type]
            prev = (*pb, float(type_id))
        else:
            prev = PREV_BBOX_NONE

        # Target element bbox
        if item_idx < len(elements):
            target_elem = elements[item_idx]
            tb = self._norm_to_retina(target_elem.bbox, pw, ph, scale, ox, oy)
            type_id = TYPE_OFFSET + ELEMENT_TYPE_TO_ID[target_elem.type]
            target = (*tb, float(type_id))
            description = target_elem.description
            target_retina = tb
        else:
            target = (0.0, 0.0, 0.0, 0.0, float(TYPE_NONE))
            description = ""
            target_retina = (0.0, 0.0, 0.0, 0.0)

        return self._pack_sample(retina, prev, target, target_retina, description)

    # -- Helpers ------------------------------------------------------------

    def _norm_to_retina(
        self,
        norm_bbox: tuple[float, float, float, float],
        img_w: int,
        img_h: int,
        scale: float,
        ox: int,
        oy: int,
    ) -> tuple[float, float, float, float]:
        """Convert normalised [0,1] bbox → pixel bbox → retina coords."""
        x1 = norm_bbox[0] * img_w
        y1 = norm_bbox[1] * img_h
        x2 = norm_bbox[2] * img_w
        y2 = norm_bbox[3] * img_h
        return (
            x1 * scale + ox,
            y1 * scale + oy,
            x2 * scale + ox,
            y2 * scale + oy,
        )

    def _pack_sample(self, retina_img, prev, target, target_retina, description):
        """Convert to tensors."""
        img_t = (
            torch.from_numpy(np.array(retina_img))
            .permute(2, 0, 1)
            .float()
            / 255.0
        )
        prev_t = torch.tensor(prev, dtype=torch.float32)
        target_t = torch.tensor(target, dtype=torch.float32)
        target_retina_t = torch.tensor(target_retina, dtype=torch.float32)

        # Tokenise description
        if description:
            encoded = self.tokenizer(
                description,
                max_length=self.max_desc_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            desc_ids = encoded["input_ids"].squeeze(0)
            desc_mask = encoded["attention_mask"].squeeze(0)
        else:
            desc_ids = torch.full((self.max_desc_len,), self.tokenizer.eos_token_id, dtype=torch.long)
            desc_mask = torch.zeros(self.max_desc_len, dtype=torch.long)

        return img_t, prev_t, target_t, desc_ids, desc_mask, target_retina_t


class PanelOnlyDataset(Dataset):
    """Subset that only yields panel-level samples (for pretraining)."""

    def __init__(self, full_dataset: MangaDetectorDataset):
        self.full = full_dataset
        self.panel_indices = [
            i for i, (_, level, _, _) in enumerate(full_dataset.index)
            if level == "panel"
        ]

    def __len__(self):
        return len(self.panel_indices)

    def __getitem__(self, idx):
        return self.full[self.panel_indices[idx]]
