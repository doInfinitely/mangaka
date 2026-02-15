"""
Format manga JSON datasets for LLM finetuning.

Exports manga pages in formats suitable for fine-tuning LLMs to generate
manga JSON from plot descriptions:

  - Chat format (OpenAI / Anthropic):  system + user plot → assistant JSON
  - Completion format:                 plot → JSON (for causal LM training)
  - Instruction format (Alpaca-style):  instruction + input → output

Each training example pairs a plot description (synthesised from the page's
own descriptions) with the full MangaPage JSON, teaching the model the
schema and visual vocabulary.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from mangaka.schema import MangaPage
from mangaka.rag.generate import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plot synthesis — turn a MangaPage's descriptions into a natural plot prompt
# ---------------------------------------------------------------------------

def _synthesise_plot(page: MangaPage) -> str:
    """Create a plausible user plot prompt from the page's own descriptions.

    This is the training target: given this plot, the model should produce
    the page's JSON.  We vary the style to avoid overfitting to one format.
    """
    templates = [
        "Create a manga page: {desc}",
        "I want a manga page showing: {desc}",
        "Design a page where {desc}",
        "Make a manga page about {desc}",
        "{desc}",
    ]

    # Combine page and panel descriptions into a narrative
    parts = []
    if page.description:
        parts.append(page.description)
    for panel in page.panels:
        if panel.description:
            parts.append(panel.description)

    desc = ". ".join(parts) if parts else "a manga page"

    template = random.choice(templates)
    return template.format(desc=desc)


def _page_to_compact_json(page: MangaPage) -> str:
    """Serialise a MangaPage to compact JSON (saves tokens in training)."""
    return json.dumps(page.model_dump(mode="json"), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Export formats
# ---------------------------------------------------------------------------

def export_chat_jsonl(
    dataset_dir: str | Path,
    output_path: str | Path,
    max_pages: int | None = None,
    seed: int = 42,
) -> int:
    """Export in OpenAI chat finetuning format (messages JSONL).

    Each line:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "plot description"},
        {"role": "assistant", "content": "MangaPage JSON"}
    ]}
    """
    random.seed(seed)
    pages = _load_pages(dataset_dir, max_pages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            plot = _synthesise_plot(page)
            page_json = _page_to_compact_json(page)
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": plot},
                    {"role": "assistant", "content": page_json},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d examples to %s (chat format)", count, output_path)
    return count


def export_completion_jsonl(
    dataset_dir: str | Path,
    output_path: str | Path,
    max_pages: int | None = None,
    seed: int = 42,
) -> int:
    """Export in completion format for causal LM finetuning.

    Each line:
    {"prompt": "...", "completion": "MangaPage JSON"}
    """
    random.seed(seed)
    pages = _load_pages(dataset_dir, max_pages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            plot = _synthesise_plot(page)
            page_json = _page_to_compact_json(page)
            example = {
                "prompt": f"### Plot\n{plot}\n\n### MangaPage JSON\n",
                "completion": page_json,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d examples to %s (completion format)", count, output_path)
    return count


def export_instruction_jsonl(
    dataset_dir: str | Path,
    output_path: str | Path,
    max_pages: int | None = None,
    seed: int = 42,
) -> int:
    """Export in Alpaca instruction format.

    Each line:
    {"instruction": "...", "input": "plot", "output": "JSON"}
    """
    random.seed(seed)
    pages = _load_pages(dataset_dir, max_pages)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for page in pages:
            plot = _synthesise_plot(page)
            page_json = _page_to_compact_json(page)
            example = {
                "instruction": (
                    "Generate a structured MangaPage JSON from the following "
                    "plot description. Include panels with bounding boxes, "
                    "element types, descriptions, and dialogue text."
                ),
                "input": plot,
                "output": page_json,
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info("Exported %d examples to %s (instruction format)", count, output_path)
    return count


# ---------------------------------------------------------------------------
# Multi-page sequence format (for models that generate whole chapters)
# ---------------------------------------------------------------------------

def export_sequence_jsonl(
    dataset_dir: str | Path,
    output_path: str | Path,
    pages_per_sequence: int = 4,
    max_sequences: int | None = None,
    seed: int = 42,
) -> int:
    """Export consecutive pages as multi-page sequences.

    Teaches the model to generate sequences of pages that form a coherent
    narrative, not just isolated pages.

    Each line:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "multi-page plot"},
        {"role": "assistant", "content": "[page1_json, page2_json, ...]"}
    ]}
    """
    random.seed(seed)
    pages = _load_pages(dataset_dir, max_pages=None)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group consecutive pages into sequences
    sequences = []
    for i in range(0, len(pages) - pages_per_sequence + 1, pages_per_sequence):
        seq = pages[i : i + pages_per_sequence]
        sequences.append(seq)

    if max_sequences:
        sequences = sequences[:max_sequences]

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for seq in sequences:
            # Synthesise a multi-page plot from the sequence
            all_parts = []
            for page in seq:
                if page.description:
                    all_parts.append(page.description)
                for panel in page.panels:
                    if panel.description:
                        all_parts.append(panel.description)
            plot = f"Create a {len(seq)}-page manga sequence: " + ". ".join(all_parts)

            pages_json = json.dumps(
                [p.model_dump(mode="json") for p in seq],
                ensure_ascii=False,
            )

            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": plot},
                    {"role": "assistant", "content": pages_json},
                ]
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    logger.info(
        "Exported %d sequences (%d pages each) to %s",
        count, pages_per_sequence, output_path,
    )
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pages(dataset_dir: str | Path, max_pages: int | None) -> list[MangaPage]:
    """Load pages from a dataset directory."""
    dataset_dir = Path(dataset_dir)
    pages_dir = dataset_dir / "pages"
    index_path = dataset_dir / "index.json"

    if index_path.exists():
        manifest = json.loads(index_path.read_text())
        file_list = manifest.get("files", [])
    else:
        file_list = [
            str(p.relative_to(pages_dir))
            for p in sorted(pages_dir.rglob("*.json"))
        ]

    pages = []
    for rel in file_list:
        if max_pages and len(pages) >= max_pages:
            break
        json_path = pages_dir / rel
        if not json_path.exists():
            continue
        try:
            pages.append(MangaPage.from_json(json_path))
        except Exception:
            continue

    return pages
