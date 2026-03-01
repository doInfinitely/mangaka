"""FastAPI application for the mangaka pipeline.

Provides REST endpoints for annotation, encoding, decoding, and generation.
Models are loaded lazily on first request to minimise startup time.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from mangaka.inference.invoke import DecodeContext, DecodeResult, MangaInvoker
from mangaka.schema import ELEMENT_TYPES, MangaPage
from mangaka.server.models import (
    AnnotateRequest,
    DatasetStatsResponse,
    DecodeRequest,
    ErrorResponse,
    GenerateRequest,
    HealthResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class _AppState:
    """Lazy-loaded application state for models and data."""

    def __init__(self) -> None:
        self.model_dir: Path = Path("checkpoints")
        self.dataset_dir: Path = Path("data/manga_dataset")
        self.detector = None
        self.tokenizer = None
        self.infiller = None
        self.index = None
        self.device = None
        self._invoker: MangaInvoker | None = None

    def get_invoker(self) -> MangaInvoker:
        """Return a MangaInvoker — Modal or local depending on env."""
        if self._invoker is None:
            if os.environ.get("MANGAKA_USE_MODAL"):
                from mangaka.inference.modal_invoker import ModalInvoker

                pool_size = int(os.environ.get("MANGAKA_MODAL_POOL_SIZE", "4"))
                self._invoker = ModalInvoker(pool_size=pool_size)
                logger.info("Using ModalInvoker (pool_size=%d)", pool_size)
            elif (self.model_dir / "controlnet_best.pt").exists() or (
                self.model_dir / "infiller"
            ).exists():
                from mangaka.inference.invoke import LocalInvoker

                self._invoker = LocalInvoker(
                    infiller_dir=str(self.model_dir),
                    detector_path=str(self.model_dir / "detector.safetensors")
                    if (self.model_dir / "detector.safetensors").exists()
                    else None,
                )
                logger.info("Using LocalInvoker")
            else:
                raise RuntimeError(
                    "No inference backend available. "
                    "Set MANGAKA_USE_MODAL=1 or provide local model weights."
                )
        return self._invoker

    def _get_device(self):
        if self.device is None:
            import torch

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self.device

    def get_detector(self):
        if self.detector is None:
            import torch
            from safetensors.torch import load_file
            from transformers import GPT2Tokenizer

            from mangaka.detector.model import MangaDetectorNet

            device = self._get_device()
            model = MangaDetectorNet(backbone="resnet50")
            ckpt = self.model_dir / "detector.safetensors"
            if ckpt.exists():
                model.load_state_dict(load_file(str(ckpt)), strict=False)
            model.to(device).eval()
            self.detector = model
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            logger.info("Detector loaded on %s", device)
        return self.detector, self.tokenizer

    def get_infiller(self):
        if self.infiller is None:
            import torch

            from mangaka.infiller.model import LayoutControlNet, MangaInfiller

            device = self._get_device()
            controlnet = LayoutControlNet()
            infiller = MangaInfiller(controlnet=controlnet)

            cn_path = self.model_dir / "controlnet.safetensors"
            if cn_path.exists():
                from safetensors.torch import load_file

                controlnet.load_state_dict(load_file(str(cn_path)), strict=False)

            infiller.load_sd_pipeline(device)

            lora_path = self.model_dir / "lora"
            if lora_path.exists():
                infiller.load_lora_weights(lora_path)

            self.infiller = infiller
            logger.info("Infiller loaded on %s", device)
        return self.infiller

    def get_index(self):
        if self.index is None:
            from mangaka.rag.index import MangaIndex

            self.index = MangaIndex()
            index_dir = self.dataset_dir.parent / "manga_index"
            if index_dir.exists():
                self.index.load(index_dir)
                logger.info("RAG index loaded from %s", index_dir)
            else:
                logger.warning("No RAG index found at %s", index_dir)
        return self.index

    def load_dataset_index(self) -> list[dict] | None:
        index_path = self.dataset_dir / "index.json"
        if index_path.exists():
            with open(index_path) as f:
                return json.load(f)
        return None


state = _AppState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Mangaka API",
    description="REST API for hierarchical manga analysis and generation.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — reports which models are loaded."""
    return HealthResponse(
        status="ok",
        detector_loaded=state.detector is not None,
        infiller_loaded=state.infiller is not None,
        index_loaded=state.index is not None,
    )


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------


@app.post("/annotate", response_model=MangaPage)
async def annotate(req: AnnotateRequest):
    """Annotate an image using a cloud VLM provider."""
    try:
        image_bytes = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data.")

    from PIL import Image

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    w, h = image.size

    if req.provider == "claude":
        from mangaka.annotator.claude import ClaudeAnnotator

        annotator = ClaudeAnnotator(two_pass=req.two_pass)
    elif req.provider == "openai":
        from mangaka.annotator.openai import OpenAIAnnotator

        annotator = OpenAIAnnotator(two_pass=req.two_pass)
    elif req.provider == "gemini":
        from mangaka.annotator.gemini import GeminiAnnotator

        annotator = GeminiAnnotator(two_pass=req.two_pass)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")

    page = await annotator.annotate(image, w, h)
    return page


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


@app.post("/encode", response_model=MangaPage)
async def encode(file: UploadFile):
    """Encode an uploaded image into a MangaPage JSON using the local detector."""
    import torch
    from PIL import Image

    from mangaka.pipeline.encode import encode_page

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")

    detector, tokenizer = state.get_detector()
    device = state._get_device()

    page = encode_page(image, detector, tokenizer, device)
    return page


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@app.post("/decode")
async def decode(req: DecodeRequest):
    """Render a MangaPage JSON into a PNG image via the invoker."""
    try:
        invoker = state.get_invoker()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    ctx = DecodeContext(
        page_json=req.page.to_json(),
        num_inference_steps=req.num_inference_steps,
        guidance_scale=req.guidance_scale,
        style_override=req.style_override,
        negative_prompt=req.negative_prompt,
        seed=req.seed,
    )
    try:
        result: DecodeResult = await asyncio.to_thread(invoker.decode, ctx)
    except Exception as exc:
        logger.exception("Decode failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return Response(content=result.image_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@app.post("/generate", response_model=list[MangaPage])
async def generate(req: GenerateRequest):
    """Generate manga page(s) from a plot description using RAG."""
    from mangaka.rag.generate import generate_manga

    index = state.get_index()
    pages = await generate_manga(
        plot=req.plot,
        index=index,
        provider=req.provider,
        num_pages=req.num_pages,
        top_k=req.top_k,
    )
    return pages


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------


@app.get("/dataset/stats", response_model=DatasetStatsResponse)
async def dataset_stats():
    """Get dataset statistics."""
    index_data = state.load_dataset_index()
    if index_data is None:
        raise HTTPException(status_code=404, detail="Dataset index.json not found.")

    type_counts: Counter[str] = Counter()
    total_panels = 0
    total_elements = 0

    for entry in index_data:
        page_path = state.dataset_dir / "pages" / entry.get("json", "")
        if not page_path.exists():
            continue
        try:
            page = MangaPage.from_json(page_path)
            total_panels += page.num_panels
            total_elements += page.num_elements
            for panel in page.panels:
                for elem in panel.elements:
                    type_counts[elem.type] += 1
        except Exception:
            continue

    return DatasetStatsResponse(
        num_pages=len(index_data),
        num_panels=total_panels,
        num_elements=total_elements,
        element_type_counts=dict(type_counts),
    )


@app.get("/dataset/pages")
async def dataset_pages(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
):
    """Paginated listing of dataset pages."""
    index_data = state.load_dataset_index()
    if index_data is None:
        raise HTTPException(status_code=404, detail="Dataset index.json not found.")

    total = len(index_data)
    items = index_data[offset : offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "items": items}


@app.get("/dataset/pages/{page_id}", response_model=MangaPage)
async def dataset_page(page_id: str):
    """Retrieve a single page JSON from the dataset."""
    page_path = state.dataset_dir / "pages" / f"{page_id}.json"
    if not page_path.exists():
        raise HTTPException(status_code=404, detail=f"Page '{page_id}' not found.")

    try:
        return MangaPage.from_json(page_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading page: {e}")
