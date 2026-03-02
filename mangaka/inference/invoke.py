"""Invoker protocol for manga encode/decode forward passes.

Defines the wire format (frozen dataclasses) and the protocol that both
LocalInvoker and ModalInvoker implement.  Following the HGI constraint:
"The mapper must be invoked, never instantiated" — all model-calling
logic flows through this interface so inference can run locally or on
a remote Modal container with no code changes.

Wire protocol: JSON strings + PNG bytes (no pickle).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from PIL import Image


# -----------------------------------------------------------------------
# Wire-format dataclasses
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class DecodeContext:
    """Everything needed to render a MangaPage JSON into an image."""
    page_json: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    style_override: str | None = None
    negative_prompt: str | None = None
    seed: int | None = None


@dataclass(frozen=True)
class DecodeResult:
    """Rendered manga page as PNG bytes."""
    image_bytes: bytes  # PNG-encoded


@dataclass(frozen=True)
class EncodeContext:
    """Everything needed to detect panels/elements from a manga image."""
    image_bytes: bytes  # PNG-encoded
    max_detections: int = 50


@dataclass(frozen=True)
class EncodeResult:
    """Detected manga page structure as JSON."""
    page_json: str


# -----------------------------------------------------------------------
# Protocol
# -----------------------------------------------------------------------

@runtime_checkable
class MangaInvoker(Protocol):
    def decode(self, ctx: DecodeContext) -> DecodeResult: ...
    def encode(self, ctx: EncodeContext) -> EncodeResult: ...


# -----------------------------------------------------------------------
# Local invoker — same-process forward pass
# -----------------------------------------------------------------------

class LocalInvoker:
    """Same-process inference. Development and testing."""

    def __init__(
        self,
        detector_path: str | None = None,
        infiller_dir: str | None = None,
        sd_model_id: str = "runwayml/stable-diffusion-inpainting",
        resolution: int = 512,
        backbone: str = "resnet50",
        gpt2_model: str = "gpt2",
        device: str = "auto",
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.sd_model_id = sd_model_id
        self.resolution = resolution
        self.backbone = backbone
        self.gpt2_model = gpt2_model

        self._detector = None
        self._tokenizer = None
        self._infiller = None

        if detector_path:
            self._load_detector(detector_path)
        if infiller_dir:
            self._load_infiller(infiller_dir)

    def _load_detector(self, path: str):
        from safetensors.torch import load_file as safetensors_load
        from transformers import GPT2Tokenizer
        from mangaka.detector.model import MangaDetectorNet

        self._detector = MangaDetectorNet(
            backbone=self.backbone, gpt2_model=self.gpt2_model,
        ).to(self.device)
        state = safetensors_load(path, device=str(self.device))
        self._detector.load_state_dict(state)
        self._detector.eval()
        self._tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model)

    def _load_infiller(self, model_dir: str):
        from pathlib import Path
        from mangaka.infiller.model import MangaInfiller, LayoutControlNet

        model_dir = Path(model_dir)

        controlnet = LayoutControlNet()
        controlnet_path = model_dir / "controlnet_best.pt"
        if controlnet_path.exists():
            controlnet.load_state_dict(
                torch.load(controlnet_path, map_location=self.device)
            )

        self._infiller = MangaInfiller(
            sd_model_id=self.sd_model_id,
            resolution=self.resolution,
            controlnet=controlnet,
        )
        self._infiller.load_sd_pipeline(self.device)

        lora_path = model_dir / "unet_lora"
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            self._infiller.load_lora_weights(lora_path)

    def decode(self, ctx: DecodeContext) -> DecodeResult:
        """Render MangaPage JSON into a PNG image."""
        if self._infiller is None:
            raise RuntimeError("Infiller not loaded. Pass infiller_dir to constructor.")

        from mangaka.schema import MangaPage

        page = MangaPage.from_json(ctx.page_json)
        image = self._infiller.render_page(
            page,
            num_inference_steps=ctx.num_inference_steps,
            guidance_scale=ctx.guidance_scale,
            style_override=ctx.style_override,
            negative_prompt=ctx.negative_prompt,
            seed=ctx.seed,
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return DecodeResult(image_bytes=buf.getvalue())

    def encode(self, ctx: EncodeContext) -> EncodeResult:
        """Detect panels/elements from a PNG image."""
        if self._detector is None or self._tokenizer is None:
            raise RuntimeError("Detector not loaded. Pass detector_path to constructor.")

        from mangaka.pipeline.encode import encode_page

        image = Image.open(io.BytesIO(ctx.image_bytes)).convert("RGB")
        page = encode_page(
            image, self._detector, self._tokenizer,
            self.device, max_detections=ctx.max_detections,
        )
        return EncodeResult(page_json=page.to_json())
