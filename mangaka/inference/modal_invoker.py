"""Modal-based invoker for distributed manga inference.

Dispatches encode/decode calls to a pool of Modal MangaWorker containers.
Uses thread-local sticky worker assignment (round-robin) to maximize
GPU cache hits — matching the glyph-daemon pattern.

Wire protocol: JSON strings + PNG bytes (no pickle).
"""

from __future__ import annotations

import threading
from itertools import cycle

import modal

from .invoke import (
    DecodeContext,
    DecodeResult,
    EncodeContext,
    EncodeResult,
    MangaInvoker,
)


class ModalInvoker:
    """Dispatches inference to Modal MangaWorker pool."""

    def __init__(self, pool_size: int = 4):
        self.pool_size = pool_size

        # Look up the deployed MangaWorker class
        self._worker_cls = modal.Cls.from_name("mangaka", "MangaWorker")

        # Create persistent worker handles for sticky assignment
        self._workers = [self._worker_cls() for _ in range(pool_size)]
        self._worker_cycle = cycle(range(pool_size))

        # Thread-local sticky assignment: each thread gets one worker
        self._local = threading.local()

    def _get_worker(self):
        """Return a sticky worker handle for the current thread."""
        if not hasattr(self._local, "worker_idx"):
            self._local.worker_idx = next(self._worker_cycle)
        return self._workers[self._local.worker_idx]

    def decode(self, ctx: DecodeContext) -> DecodeResult:
        """Send decode request to a Modal worker and return rendered image."""
        worker = self._get_worker()

        image_bytes = worker.decode.remote(
            ctx.page_json,
            ctx.num_inference_steps,
            ctx.guidance_scale,
            ctx.style_override,
            ctx.negative_prompt,
            ctx.seed,
        )

        return DecodeResult(image_bytes=image_bytes)

    def encode(self, ctx: EncodeContext) -> EncodeResult:
        """Send encode request to a Modal worker and return detected JSON."""
        worker = self._get_worker()

        page_json = worker.encode.remote(
            ctx.image_bytes,
            ctx.max_detections,
        )

        return EncodeResult(page_json=page_json)
