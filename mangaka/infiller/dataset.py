"""
Dataset for training the MangaInfiller.

Produces training samples at every level of the containment hierarchy,
mirroring how the infiller operates at inference time:

  Page-level samples:   full page image + panel mask + page layout + panel description
  Panel-level samples:  panel crop image + element mask + panel layout + element description
  Element-level samples: element crop image + full mask + element layout + element description

Each level gives the model the full SD resolution for the target region,
teaching it to infill at every scale of the hierarchy.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from mangaka.schema import MangaPage, Panel, MangaElement
from mangaka.infiller.layout import (
    render_page_layout,
    render_panel_layout,
    render_element_layout,
    render_panel_mask,
    render_element_mask_in_panel,
    render_full_mask,
)
from mangaka.utils.bbox import denormalise_bbox

logger = logging.getLogger(__name__)


class MangaInfillerDataset(Dataset):
    """Hierarchy-aware dataset for SD infiller training.

    Each sample is drawn from one of three hierarchy levels:
      - Page level:    (page_image_resized, panel_mask, page_layout, panel_desc)
      - Panel level:   (panel_crop_resized, element_mask, panel_layout, elem_desc)
      - Element level: (element_crop_resized, full_mask, element_layout, elem_desc)

    Returns (image, mask, layout, prompt_token_ids) for each sample.
    The model learns to infill at every scale.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_dir: str | Path,
        resolution: int = 512,
        tokenizer: CLIPTokenizer | None = None,
        max_prompt_length: int = 77,
        element_level_types: tuple[str, ...] = ("character", "sfx"),
    ):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.resolution = resolution
        self.max_prompt_length = max_prompt_length
        self.element_level_types = element_level_types

        if tokenizer is None:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        else:
            self.tokenizer = tokenizer

        # Discover pairs and build hierarchical index
        self.pages: list[tuple[Path, MangaPage]] = []
        self.index: list[tuple[int, str, int, int]] = []
        # index entries: (page_idx, level, panel_idx, element_idx)
        #   level = "page"    → panel_idx is the target panel, element_idx = -1
        #   level = "panel"   → panel_idx is parent, element_idx is target element
        #   level = "element" → panel_idx is parent, element_idx is target element
        self._discover_and_index()

    def _discover_and_index(self):
        json_paths = sorted(self.annotation_dir.rglob("*.json"))
        for json_path in json_paths:
            rel = json_path.relative_to(self.annotation_dir)
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                candidate = self.image_dir / rel.with_suffix(ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            try:
                page = MangaPage.from_json(json_path)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", json_path, e)
                continue

            if not page.panels:
                continue

            page_idx = len(self.pages)
            self.pages.append((img_path, page))

            for pi, panel in enumerate(page.panels):
                # Page-level sample: infill this panel
                self.index.append((page_idx, "page", pi, -1))

                for ei, elem in enumerate(panel.elements):
                    # Panel-level sample: infill this element within the panel
                    self.index.append((page_idx, "panel", pi, ei))

                    # Element-level sample (only for types that benefit from zoom)
                    if elem.type in self.element_level_types:
                        self.index.append((page_idx, "element", pi, ei))

        logger.info(
            "Indexed %d pages → %d hierarchy samples "
            "(page/panel/element levels)",
            len(self.pages), len(self.index),
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        page_idx, level, panel_idx, element_idx = self.index[idx]
        img_path, page = self.pages[page_idx]

        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        if level == "page":
            return self._make_page_sample(image, page, panel_idx)
        elif level == "panel":
            return self._make_panel_sample(image, page, panel_idx, element_idx)
        else:  # element
            return self._make_element_sample(image, page, panel_idx, element_idx)

    # -- Page-level: full page → infill a panel -----------------------------

    def _make_page_sample(self, image: Image.Image, page: MangaPage, panel_idx: int):
        """Page at full res, mask = panel region, prompt = panel description."""
        res = self.resolution
        panel = page.panels[panel_idx]

        # Resize full page to SD resolution
        img_res = image.resize((res, res), Image.LANCZOS)
        img_np = np.array(img_res).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Page-level layout
        layout = render_page_layout(page, res)
        layout_tensor = torch.from_numpy(layout)

        # Panel mask at page scale
        w, h = page.width, page.height
        mask = np.zeros((1, res, res), dtype=np.float32)
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
        sx, sy = res / w, res / h
        mx1 = max(0, int(px1 * sx))
        my1 = max(0, int(py1 * sy))
        mx2 = min(res, int(px2 * sx))
        my2 = min(res, int(py2 * sy))
        mask[0, my1:my2, mx1:mx2] = 1.0
        mask_tensor = torch.from_numpy(mask)

        prompt = panel.description
        input_ids = self._tokenize(prompt)

        return img_tensor, mask_tensor, layout_tensor, input_ids

    # -- Panel-level: panel crop → infill an element ------------------------

    def _make_panel_sample(
        self, image: Image.Image, page: MangaPage, panel_idx: int, element_idx: int,
    ):
        """Panel crop at full res, mask = element region, prompt = element desc."""
        res = self.resolution
        w, h = image.size
        panel = page.panels[panel_idx]
        elem = panel.elements[element_idx]

        # Crop panel from the page
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        panel_crop = image.crop((px1, py1, px2, py2))

        # Resize panel crop to SD resolution
        panel_res = panel_crop.resize((res, res), Image.LANCZOS)
        img_np = np.array(panel_res).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Panel-level layout (element borders + types)
        layout = render_panel_layout(panel, res)
        layout_tensor = torch.from_numpy(layout)

        # Element mask within the panel
        mask = render_element_mask_in_panel(panel, element_idx, res)
        mask_tensor = torch.from_numpy(mask)

        prompt = elem.description
        if elem.text:
            prompt += f' with text "{elem.text}"'
        input_ids = self._tokenize(prompt)

        return img_tensor, mask_tensor, layout_tensor, input_ids

    # -- Element-level: element crop → full refinement ----------------------

    def _make_element_sample(
        self, image: Image.Image, page: MangaPage, panel_idx: int, element_idx: int,
    ):
        """Element crop at full res, full mask, prompt = element desc."""
        res = self.resolution
        w, h = image.size
        panel = page.panels[panel_idx]
        elem = panel.elements[element_idx]

        # Crop panel, then crop element from panel
        px1, py1, px2, py2 = denormalise_bbox(panel.bbox, w, h)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w, px2), min(h, py2)
        pw, ph = px2 - px1, py2 - py1

        ex1 = max(0, int(px1 + elem.bbox[0] * pw))
        ey1 = max(0, int(py1 + elem.bbox[1] * ph))
        ex2 = min(w, int(px1 + elem.bbox[2] * pw))
        ey2 = min(h, int(py1 + elem.bbox[3] * ph))
        elem_crop = image.crop((ex1, ey1, ex2, ey2))

        # Resize element crop to SD resolution
        elem_res = elem_crop.resize((res, res), Image.LANCZOS)
        img_np = np.array(elem_res).astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Element-level layout (full type fill)
        layout = render_element_layout(elem, res)
        layout_tensor = torch.from_numpy(layout)

        # Full mask (refine everything)
        mask = render_full_mask(res)
        mask_tensor = torch.from_numpy(mask)

        prompt = elem.description
        if elem.text:
            prompt += f' with text "{elem.text}"'
        input_ids = self._tokenize(prompt)

        return img_tensor, mask_tensor, layout_tensor, input_ids

    # -- Helpers ------------------------------------------------------------

    def _tokenize(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_prompt_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokens["input_ids"].squeeze(0)
