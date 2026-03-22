"""
Dataset builder: batch-encode a manga image directory into a JSON dataset.

Produces:
  output_dir/
    index.json          — dataset metadata and manifest
    pages/
      vol01/ch01/page001.json
      vol01/ch01/page002.json
      ...

The index.json contains statistics and a file list for easy consumption
by RAG systems and LLM finetuning pipelines.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from mangaka.schema import MangaPage

logger = logging.getLogger(__name__)
console = Console()

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def build_dataset(
    image_dir: str,
    output_dir: str,
    model_path: str | None = None,
    annotation_dir: str | None = None,
    backbone: str = "resnet50",
    lm_model: str = "Qwen/Qwen2.5-0.5B",
    max_detections: int = 50,
) -> dict:
    """Build a JSON manga dataset from images.

    If *annotation_dir* is provided, uses existing annotations.
    Otherwise, runs the detector to generate annotations.

    Args:
        image_dir: directory containing manga images
        output_dir: directory to write the JSON dataset
        model_path: path to trained detector (required if no annotation_dir)
        annotation_dir: path to existing annotations (skips detection)
        backbone: detector backbone architecture
        lm_model: causal LM variant for description generation
        max_detections: max detections per hierarchy level

    Returns:
        dict with dataset statistics
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    pages_dir = output_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    stats = {
        "total_pages": 0,
        "total_panels": 0,
        "total_elements": 0,
        "element_type_counts": {},
        "files": [],
        "errors": 0,
    }

    if annotation_dir:
        # Use existing annotations
        _build_from_annotations(annotation_dir, pages_dir, stats)
    elif model_path:
        # Generate annotations with detector
        _build_from_detector(
            image_dir, pages_dir, model_path, backbone,
            lm_model, max_detections, stats,
        )
    else:
        raise ValueError("Either annotation_dir or model_path must be provided")

    # Write index
    elapsed = time.time() - start_time
    index = {
        "version": "1.0",
        "source_dir": str(image_dir),
        "total_pages": stats["total_pages"],
        "total_panels": stats["total_panels"],
        "total_elements": stats["total_elements"],
        "element_type_counts": stats["element_type_counts"],
        "errors": stats["errors"],
        "build_time_seconds": round(elapsed, 1),
        "files": stats["files"],
    }

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    console.print(f"\n[green]Dataset built![/green]")
    console.print(f"  Pages: {stats['total_pages']}")
    console.print(f"  Panels: {stats['total_panels']}")
    console.print(f"  Elements: {stats['total_elements']}")
    console.print(f"  Errors: {stats['errors']}")
    console.print(f"  Time: {elapsed:.1f}s")
    console.print(f"  Output: {output_dir}")

    return index


def _build_from_annotations(annotation_dir: str, pages_dir: Path, stats: dict):
    """Copy and index existing annotations."""
    ann_dir = Path(annotation_dir)
    json_paths = sorted(ann_dir.rglob("*.json"))
    console.print(f"Indexing {len(json_paths)} existing annotations...")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Building...", total=len(json_paths))

        for json_path in json_paths:
            try:
                page = MangaPage.from_json(json_path)
                _index_page(page, json_path, ann_dir, pages_dir, stats)
                progress.update(task, advance=1)
            except Exception as e:
                logger.error("Failed to process %s: %s", json_path, e)
                stats["errors"] += 1
                progress.update(task, advance=1)


def _build_from_detector(
    image_dir: Path, pages_dir: Path, model_path: str,
    backbone: str, lm_model: str, max_detections: int, stats: dict,
):
    """Generate annotations using the trained detector."""
    import torch
    import numpy as np
    from PIL import Image
    from safetensors.torch import load_file as safetensors_load
    from transformers import AutoTokenizer
    from mangaka.detector.model import MangaDetectorNet
    from mangaka.pipeline.encode import encode_page

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MangaDetectorNet(backbone=backbone, lm_model=lm_model).to(device)
    state = safetensors_load(model_path, device=str(device))
    model.load_state_dict(state)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(lm_model)

    image_paths = sorted(
        p for p in image_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file()
    )

    console.print(f"Encoding {len(image_paths)} images with detector...")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Encoding...", total=len(image_paths))

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                page = encode_page(image, model, tokenizer, device, max_detections)

                # Save JSON
                rel = img_path.relative_to(image_dir)
                json_path = pages_dir / rel.with_suffix(".json")
                json_path.parent.mkdir(parents=True, exist_ok=True)
                page.to_json(json_path)

                _update_stats(page, str(rel.with_suffix(".json")), stats)
                progress.update(task, advance=1)
            except Exception as e:
                logger.error("Failed to encode %s: %s", img_path, e)
                stats["errors"] += 1
                progress.update(task, advance=1)


def _index_page(
    page: MangaPage, json_path: Path, ann_dir: Path,
    pages_dir: Path, stats: dict,
):
    """Copy annotation to pages_dir and update stats."""
    import shutil

    rel = json_path.relative_to(ann_dir)
    dest = pages_dir / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Re-save (ensures consistent formatting)
    page.to_json(dest)

    _update_stats(page, str(rel), stats)


def _update_stats(page: MangaPage, rel_path: str, stats: dict):
    """Update running statistics."""
    stats["total_pages"] += 1
    stats["total_panels"] += page.num_panels
    stats["total_elements"] += page.num_elements
    stats["files"].append(rel_path)

    for panel in page.panels:
        for elem in panel.elements:
            key = elem.type
            stats["element_type_counts"][key] = (
                stats["element_type_counts"].get(key, 0) + 1
            )
