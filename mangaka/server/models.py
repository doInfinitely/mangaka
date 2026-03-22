"""Request / response Pydantic models for the mangaka REST API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mangaka.schema import MangaPage


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class AnnotateRequest(BaseModel):
    """Request to annotate a manga image via a cloud VLM provider."""

    provider: Literal["claude", "openai", "gemini"] = Field(
        default="claude",
        description="VLM provider to use for annotation.",
    )
    image_base64: str = Field(
        ...,
        description="Base64-encoded image data (PNG or JPEG).",
    )
    two_pass: bool = Field(
        default=True,
        description="Use two-pass annotation (detection + description).",
    )


class GenerateRequest(BaseModel):
    """Request to generate manga pages from a plot description."""

    plot: str = Field(..., description="Plot/story description.")
    num_pages: int = Field(default=1, ge=1, le=20, description="Number of pages to generate.")
    provider: Literal["claude", "openai", "gemini"] = Field(
        default="claude",
        description="LLM provider for generation.",
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of RAG examples to retrieve.")
    style: str | None = Field(
        default=None,
        description="Art style override for rendering.",
    )


class DecodeRequest(BaseModel):
    """Request to render a MangaPage JSON into an image."""

    page: MangaPage = Field(..., description="MangaPage JSON to render.")
    num_inference_steps: int = Field(default=50, ge=1, le=200)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    style_override: str | None = Field(default=None)
    negative_prompt: str | None = Field(default=None)
    seed: int | None = Field(default=None)
    artist_names: list[str] | None = Field(
        default=None,
        description="Artist names for post-infill style transfer.",
    )


class PrepareStyleRequest(BaseModel):
    """Request to fetch galleries and encode style embeddings for artists."""

    artists: list[str] = Field(..., description="Artist names to prepare style banks for.")
    cache_dir: str | None = Field(default=None, description="Gallery cache directory override.")
    max_galleries: int = Field(default=20, ge=1, le=100, description="Max galleries per artist.")


class PrepareStyleResponse(BaseModel):
    """Status of style preparation per artist."""

    status: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-artist status: { artist: { images: N, embeddings: N } }",
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Server health check response."""

    status: str = "ok"
    detector_loaded: bool = False
    infiller_loaded: bool = False
    index_loaded: bool = False


class DatasetStatsResponse(BaseModel):
    """Dataset statistics."""

    num_pages: int = 0
    num_panels: int = 0
    num_elements: int = 0
    element_type_counts: dict[str, int] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
