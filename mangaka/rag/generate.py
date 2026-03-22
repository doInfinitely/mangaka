"""
RAG-driven manga JSON generation.

Given a user's plot description:
  1. Retrieve relevant manga pages from the index
  2. Build a prompt with retrieved examples as few-shot context
  3. Call an LLM to generate a new MangaPage JSON skeleton
  4. Validate and return the structured result

The retrieved examples teach the LLM the JSON schema, typical panel/element
patterns, and the visual vocabulary — so it produces manga JSONs that are
consistent with the dataset and ready for the infiller to render.
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from mangaka.schema import MangaPage
from mangaka.rag.index import MangaIndex
from mangaka.annotator.prompt import _extract_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for manga JSON generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a manga creator. You design manga pages as structured JSON.

Each page has panels arranged in reading order. Each panel has a description
and a list of visual elements (characters, speech bubbles, sound effects,
backgrounds, effects, objects).

Bounding boxes are normalised to [0, 1]:
  - Panel bboxes are relative to the page
  - Element bboxes are relative to their parent panel

Element types: character, speech_bubble, sfx, background, effect, object

The page has an optional "style" field (string) that describes the art style
for rendering, e.g. "manga, black and white, ink drawing, screentone shading".
If the user specifies an art style, set this field accordingly.

You must return ONLY valid JSON matching the MangaPage schema."""


def _build_generation_prompt(
    plot: str,
    examples: list[MangaPage],
    page_number: int = 1,
    total_pages: int = 1,
    previous_pages: list[MangaPage] | None = None,
    page_spec: dict | None = None,
) -> str:
    """Build the user prompt for generating a single page.

    Each page is generated one at a time so it can be conditioned on
    the preceding pages for narrative continuity.
    """

    parts = []

    # Show RAG examples
    if examples:
        parts.append(
            f"Here are {len(examples)} example manga pages for reference. "
            "Study the structure, element placement, and description style:\n"
        )
        for i, ex in enumerate(examples):
            ex_json = ex.model_dump(mode="json")
            parts.append(f"--- Example {i + 1} ---")
            parts.append(json.dumps(ex_json, indent=None, ensure_ascii=False))
            parts.append("")

    # Show previously generated pages for continuity
    if previous_pages:
        parts.append(
            f"--- Story so far ({len(previous_pages)} page(s) already created) ---"
        )
        for i, prev in enumerate(previous_pages):
            prev_json = prev.model_dump(mode="json")
            parts.append(f"Page {i + 1}:")
            parts.append(json.dumps(prev_json, indent=None, ensure_ascii=False))
            parts.append("")
        parts.append(
            "Continue the story naturally from where the previous page(s) left off. "
            "Maintain character consistency, advance the plot, and vary panel compositions.\n"
        )

    # User request
    parts.append("--- Your task ---")
    parts.append(
        f"Create page {page_number} of {total_pages} for the following plot:\n"
    )
    parts.append(f"Plot: {plot}\n")

    if page_spec:
        if page_spec.get("panel_count"):
            parts.append(f"Target panel count per page: {page_spec['panel_count']}")
        if page_spec.get("width") and page_spec.get("height"):
            parts.append(f"Page dimensions: {page_spec['width']}x{page_spec['height']}")
        if page_spec.get("style"):
            parts.append(f'Art style: {page_spec["style"]}')
            parts.append(f'Set the "style" field in the JSON to: "{page_spec["style"]}"')

    parts.append(
        "\nReturn a single JSON object matching the MangaPage schema."
    )

    parts.append(
        "\nMake bounding boxes tight and non-overlapping. "
        "Write vivid, specific descriptions. "
        "Include dialogue in speech_bubble text fields."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generation backends
# ---------------------------------------------------------------------------

async def _call_claude(prompt: str, system: str) -> str:
    import os
    import anthropic
    client = anthropic.AsyncAnthropic(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )
    resp = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


async def _call_openai(prompt: str, system: str) -> str:
    import os
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = await client.chat.completions.create(
        model="gpt-4o",
        max_tokens=8192,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


async def _call_gemini(prompt: str, system: str) -> str:
    import os
    from google import genai
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    resp = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=[f"{system}\n\n{prompt}"],
    )
    return resp.text


async def _call_llama(prompt: str, system: str) -> str:
    import os
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        api_key=os.environ.get("LLAMA_API_KEY"),
        base_url=os.environ.get("LLAMA_BASE_URL", "https://api.together.xyz/v1"),
    )
    resp = await client.chat.completions.create(
        model=os.environ.get("LLAMA_MODEL", "meta-llama/Llama-Vision-Free"),
        max_tokens=8192,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


_BACKENDS = {
    "llama": _call_llama,
    "claude": _call_claude,
    "openai": _call_openai,
    "gemini": _call_gemini,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate_manga(
    plot: str,
    index: MangaIndex,
    provider: Literal["llama", "claude", "openai", "gemini"] = "llama",
    num_pages: int = 1,
    top_k: int = 5,
    page_spec: dict | None = None,
    structure_hint: dict | None = None,
) -> list[MangaPage]:
    """Generate manga page JSON(s) using RAG.

    1. Retrieve top_k relevant pages from the index
    2. Build a prompt with examples + user plot
    3. Call the LLM to generate new MangaPage JSON
    4. Validate and return

    Args:
        plot: user's plot / story description
        index: loaded MangaIndex
        provider: which LLM backend to use
        num_pages: number of pages to generate
        top_k: number of examples to retrieve
        page_spec: optional page constraints (panel_count, width, height, style)
        structure_hint: optional structural bias for retrieval

    Returns:
        list of validated MangaPage objects
    """
    # Step 1: retrieve RAG examples (skip if index not built)
    examples: list[MangaPage] = []
    try:
        results = index.query(plot, top_k=top_k, structure_hint=structure_hint)
        examples = [page for page, _, _ in results]
        if results:
            logger.info(
                "Retrieved %d examples (top score: %.3f)",
                len(results), results[0][2],
            )
    except RuntimeError:
        logger.warning("RAG index not available, generating without examples.")

    backend = _BACKENDS[provider]
    pages: list[MangaPage] = []

    # Generate pages sequentially, each conditioned on the previous ones
    for page_num in range(1, num_pages + 1):
        logger.info("Generating page %d of %d...", page_num, num_pages)

        prompt = _build_generation_prompt(
            plot,
            examples,
            page_number=page_num,
            total_pages=num_pages,
            previous_pages=pages if pages else None,
            page_spec=page_spec,
        )

        raw = await backend(prompt, SYSTEM_PROMPT)
        new_pages = _parse_generation_response(raw, 1)

        if new_pages:
            pages.append(new_pages[0])
        else:
            logger.warning("Failed to generate page %d, skipping", page_num)

    logger.info("Generated %d page(s)", len(pages))
    return pages


def _parse_generation_response(raw: str, expected: int) -> list[MangaPage]:
    """Parse LLM response into MangaPage objects."""
    data = _extract_json(raw)

    # Could be a single page dict or an array
    if isinstance(data, list):
        page_dicts = data
    elif isinstance(data, dict):
        # Check if it has a "pages" key (some models wrap it)
        if "pages" in data and isinstance(data["pages"], list):
            page_dicts = data["pages"]
        else:
            page_dicts = [data]
    else:
        raise ValueError(f"Unexpected response type: {type(data)}")

    pages = []
    for d in page_dicts:
        try:
            # Provide defaults for width/height if missing
            d.setdefault("width", 1654)
            d.setdefault("height", 2339)
            d.setdefault("description", "")
            # Fix elements missing description (LLMs often omit it for speech_bubble/sfx)
            for panel in d.get("panels", []):
                for elem in panel.get("elements", []):
                    if "description" not in elem:
                        elem["description"] = elem.get("text", "")
            page = MangaPage.model_validate(d)
            pages.append(page)
        except Exception as e:
            logger.warning("Failed to validate page: %s", e)

    return pages
