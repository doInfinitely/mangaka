"""Training dataset for the style transfer GAN.

Creates (content, target, style_bank) triplets:
    - target: an artist's original image
    - content: edge-extracted version (structure without style)
    - style_bank: pre-computed style embeddings for that artist
      (excluding the target image to prevent trivial copying)
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms

# Allow loading of slightly truncated images rather than crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def extract_edges(image: Image.Image) -> Image.Image:
    """Extract structural edges from an image (style-free content).

    Applies a Canny-like edge detection pipeline using PIL filters,
    producing a 3-channel edge map suitable as content input.
    """
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Dilate slightly for cleaner lines
    edges = edges.filter(ImageFilter.MaxFilter(3))
    # Back to 3-channel
    return Image.merge("RGB", [edges, edges, edges])


class StyleTransferDataset(Dataset):
    """Dataset for style transfer GAN training.

    Directory layout expected:
        image_dir/
            artist_001/
                img_0001.png
                img_0002.png
                ...
            artist_002/
                ...

        bank_dir/
            artist_001.pt  # (N, style_dim) pre-computed embeddings
            artist_002.pt
            ...

    Each sample returns:
        - content: (3, H, W) edge-extracted input in [-1, 1]
        - target: (3, H, W) original image in [-1, 1]
        - style_bank: (N, style_dim) all embeddings for the artist
          (with the target image's embedding removed)
        - target_idx: index of the target image within the artist's folder
          (used to exclude it from the bank)
    """

    def __init__(
        self,
        image_dir: str | Path,
        bank_dir: str | Path,
        resolution: int = 512,
        max_bank_size: int = 256,
    ):
        self.image_dir = Path(image_dir)
        self.bank_dir = Path(bank_dir)
        self.resolution = resolution
        self.max_bank_size = max_bank_size

        # Build sample index: list of (artist_id, image_path, image_idx)
        self.samples: list[tuple[str, Path, int]] = []
        self._bank_cache: dict[str, torch.Tensor] = {}

        for artist_dir in sorted(self.image_dir.iterdir()):
            if not artist_dir.is_dir():
                continue
            artist_id = artist_dir.name
            bank_path = self.bank_dir / f"{artist_id}.pt"
            if not bank_path.exists():
                logger.warning("No bank for artist '%s', skipping", artist_id)
                continue

            images = sorted(
                p for p in artist_dir.iterdir()
                if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
            )
            for idx, img_path in enumerate(images):
                self.samples.append((artist_id, img_path, idx))

        logger.info("StyleTransferDataset: %d samples from %d artists",
                     len(self.samples), len(set(s[0] for s in self.samples)))

        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),            # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
        ])

        self.content_transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def _load_bank(self, artist_id: str) -> torch.Tensor:
        if artist_id not in self._bank_cache:
            path = self.bank_dir / f"{artist_id}.pt"
            self._bank_cache[artist_id] = torch.load(
                path, map_location="cpu", weights_only=True
            )
        return self._bank_cache[artist_id]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        artist_id, img_path, img_idx = self.samples[idx]

        # Load and transform target image — retry with a random fallback on corrupt files
        try:
            image = Image.open(img_path)
            image.load()  # force full decode to catch truncated files
            image = image.convert("RGB")
        except (OSError, SyntaxError):
            logger.warning("Corrupt image %s, using random fallback", img_path)
            fallback_idx = (idx + 1) % len(self.samples)
            return self[fallback_idx]

        target = self.transform(image)

        # Extract edges for content input
        edges = extract_edges(image)
        content = self.content_transform(edges)

        # Load style bank, excluding this image's embedding
        full_bank = self._load_bank(artist_id)
        mask = torch.ones(full_bank.shape[0], dtype=torch.bool)
        if img_idx < full_bank.shape[0]:
            mask[img_idx] = False
        bank = full_bank[mask]

        # Subsample if bank is too large
        if bank.shape[0] > self.max_bank_size:
            indices = torch.randperm(bank.shape[0])[: self.max_bank_size]
            bank = bank[indices]

        return {
            "content": content,
            "target": target,
            "style_bank": bank,
            "artist_id": artist_id,
        }


def collate_style_batch(batch: list[dict]) -> dict[str, torch.Tensor | list]:
    """Custom collate that pads style banks to the same length within a batch."""
    contents = torch.stack([b["content"] for b in batch])
    targets = torch.stack([b["target"] for b in batch])
    artist_ids = [b["artist_id"] for b in batch]

    # Pad banks to max length in batch
    banks = [b["style_bank"] for b in batch]
    max_n = max(b.shape[0] for b in banks)
    style_dim = banks[0].shape[1]

    padded = torch.zeros(len(banks), max_n, style_dim)
    bank_masks = torch.zeros(len(banks), max_n, dtype=torch.bool)

    for i, b in enumerate(banks):
        n = b.shape[0]
        padded[i, :n] = b
        bank_masks[i, :n] = True

    return {
        "content": contents,
        "target": targets,
        "style_bank": padded,
        "bank_mask": bank_masks,
        "artist_ids": artist_ids,
    }
