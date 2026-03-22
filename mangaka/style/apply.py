"""Post-infill style applicator — wraps the generator + bank for inference.

Usage:
    applicator = StyleApplicator("data/style_bank", "checkpoints/style/generator_best.pt")
    styled = applicator.stylize(raw_image, ["artist_a", "artist_b"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

logger = logging.getLogger(__name__)


class StyleApplicator:
    """Apply style transfer post-infill using the trained generator + style bank.

    Lazy-loads the generator checkpoint and style bank on first use to avoid
    wasting GPU memory when no artists are selected.
    """

    def __init__(
        self,
        bank_dir: str | Path,
        generator_path: str | Path,
        style_dim: int = 512,
        device: str = "auto",
    ):
        self.bank_dir = Path(bank_dir)
        self.generator_path = Path(generator_path)
        self.style_dim = style_dim

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._generator = None
        self._bank = None

    def _load_generator(self):
        """Lazy-load the StyleTransferGenerator from checkpoint."""
        if self._generator is not None:
            return

        from mangaka.style.model import StyleTransferGenerator

        if not self.generator_path.exists():
            logger.warning("Generator checkpoint not found at %s", self.generator_path)
            return

        self._generator = StyleTransferGenerator(style_dim=self.style_dim).to(self.device)
        state = torch.load(self.generator_path, map_location=self.device, weights_only=True)
        self._generator.load_state_dict(state)
        self._generator.float()  # ensure fp32 — checkpoint may have fp16 from AMP training
        self._generator.eval()
        logger.info("StyleTransferGenerator loaded from %s", self.generator_path)

    def _load_bank(self):
        """Lazy-load the StyleBank."""
        if self._bank is not None:
            return

        from mangaka.style.bank import StyleBank

        self._bank = StyleBank(self.bank_dir, style_dim=self.style_dim)

    def clear_bank_cache(self):
        """Clear cached style bank embeddings (forces re-read from disk)."""
        if self._bank is not None:
            self._bank.clear_cache()

    @torch.no_grad()
    def stylize(
        self,
        image: Image.Image,
        artist_names: list[str],
        resolution: int = 512,
    ) -> Image.Image:
        """Apply style transfer to a PIL image.

        Args:
            image: Raw infilled page (PIL RGB).
            artist_names: List of artist IDs whose banks to merge.
            resolution: Resolution for the style transfer pass.

        Returns:
            Styled PIL image (same size as input). Returns the original
            image unchanged if no bank is available or generator is missing.
        """
        if not artist_names:
            return image

        self._load_generator()
        self._load_bank()

        if self._generator is None or self._bank is None:
            return image

        # Collect embeddings from all requested artists
        all_embeddings: list[torch.Tensor] = []
        for name in artist_names:
            try:
                emb = self._bank.load_artist(name, device=self.device)
                all_embeddings.append(emb)
            except FileNotFoundError:
                logger.warning("No style bank for artist '%s', skipping", name)

        if not all_embeddings:
            return image

        # Concatenate all artist embeddings into a single bank: (1, N_total, style_dim)
        bank = torch.cat(all_embeddings, dim=0).unsqueeze(0).float()

        # Prepare content tensor — edge-extract first to match training domain
        from mangaka.style.dataset import extract_edges

        orig_size = image.size  # (W, H)
        edges = extract_edges(image.convert("RGB"))
        transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.LANCZOS,
            ),
            transforms.ToTensor(),            # [0, 1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1, 1]
        ])
        content = transform(edges).unsqueeze(0).to(self.device)

        styled = self._generator(content, bank)  # (1, 3, H, W) in [-1, 1]

        # Convert back to PIL
        styled = styled.squeeze(0).clamp(-1, 1)
        styled = (styled + 1.0) / 2.0  # back to [0, 1]
        result = to_pil_image(styled.cpu())

        # Resize back to original dimensions
        if result.size != orig_size:
            result = result.resize(orig_size, Image.LANCZOS)

        return result
