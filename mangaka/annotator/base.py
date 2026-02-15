"""
Abstract base class for cloud LLM manga annotators.

Every provider (Claude, OpenAI, Gemini) implements this interface.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import io
import logging
from pathlib import Path

from PIL import Image

from mangaka.schema import MangaPage

logger = logging.getLogger(__name__)


def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL Image as a base64 data string."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk, converting to RGB."""
    return Image.open(path).convert("RGB")


class Annotator(abc.ABC):
    """Base class for cloud LLM manga page annotators.

    Subclasses implement :meth:`_annotate_page_impl` which sends a single
    image to the cloud API and returns a :class:`MangaPage`.

    The public :meth:`annotate` and :meth:`annotate_batch` methods add
    retry logic, rate limiting, and concurrency control.
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        two_pass: bool = True,
        crop_padding: float = 0.05,
    ):
        self.max_concurrency = max_concurrency
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.two_pass = two_pass
        self.crop_padding = crop_padding

    # -- Abstract methods subclasses must implement -------------------------

    @abc.abstractmethod
    async def _call_vlm(
        self,
        image: Image.Image,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Send an image + text prompt to the cloud VLM, return raw text."""
        ...

    # -- Public API ---------------------------------------------------------

    async def annotate(self, image: Image.Image, width: int, height: int) -> MangaPage:
        """Annotate a single manga page image.

        If *two_pass* is enabled, first detects layout (panels + element
        bboxes), then crops and re-sends each element for detailed
        descriptions.
        """
        from mangaka.annotator.prompt import (
            DETECTION_PROMPT,
            DETECTION_SYSTEM,
            DESCRIPTION_PROMPT,
            DESCRIPTION_SYSTEM,
            parse_detection_response,
            parse_description_response,
        )

        # Pass 1: detection
        raw = await self._call_with_retry(image, DETECTION_PROMPT, DETECTION_SYSTEM)
        page = parse_detection_response(raw, width, height)

        if not self.two_pass:
            return page

        # Pass 2: element-level descriptions via crops
        for panel in page.panels:
            pw = int(panel.bbox[2] * width) - int(panel.bbox[0] * width)
            ph = int(panel.bbox[3] * height) - int(panel.bbox[1] * height)
            px1 = int(panel.bbox[0] * width)
            py1 = int(panel.bbox[1] * height)

            for elem in panel.elements:
                # Element bbox is relative to panel
                ex1 = px1 + int(elem.bbox[0] * pw)
                ey1 = py1 + int(elem.bbox[1] * ph)
                ex2 = px1 + int(elem.bbox[2] * pw)
                ey2 = py1 + int(elem.bbox[3] * ph)

                # Add padding
                pad_x = int((ex2 - ex1) * self.crop_padding)
                pad_y = int((ey2 - ey1) * self.crop_padding)
                cx1 = max(0, ex1 - pad_x)
                cy1 = max(0, ey1 - pad_y)
                cx2 = min(width, ex2 + pad_x)
                cy2 = min(height, ey2 + pad_y)

                crop = image.crop((cx1, cy1, cx2, cy2))
                desc_prompt = DESCRIPTION_PROMPT.format(element_type=elem.type)
                raw_desc = await self._call_with_retry(
                    crop, desc_prompt, DESCRIPTION_SYSTEM,
                )
                parsed = parse_description_response(raw_desc)
                if parsed:
                    elem.description = parsed

        return page

    async def annotate_batch(
        self,
        images: list[tuple[Image.Image, int, int]],
    ) -> list[MangaPage]:
        """Annotate multiple pages with concurrency control.

        *images* is a list of (image, width, height) tuples.
        """
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _limited(img: Image.Image, w: int, h: int) -> MangaPage:
            async with sem:
                return await self.annotate(img, w, h)

        tasks = [_limited(img, w, h) for img, w, h in images]
        return await asyncio.gather(*tasks)

    # -- Internal helpers ---------------------------------------------------

    async def _call_with_retry(
        self,
        image: Image.Image,
        prompt: str,
        system: str | None = None,
    ) -> str:
        """Call the VLM with exponential-backoff retries."""
        last_err: Exception | None = None
        for attempt in range(self.retry_attempts):
            try:
                return await self._call_vlm(image, prompt, system)
            except Exception as e:
                last_err = e
                delay = self.retry_delay * (2 ** attempt)
                logger.warning(
                    "Attempt %d/%d failed: %s  (retrying in %.1fs)",
                    attempt + 1, self.retry_attempts, e, delay,
                )
                await asyncio.sleep(delay)
        raise RuntimeError(
            f"All {self.retry_attempts} attempts failed"
        ) from last_err
