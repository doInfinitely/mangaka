"""
Distillation loop — iteratively improve the local detector using a cloud LLM teacher.

Strategy:
  1. Bootstrap: Start with detector trained on initial cloud annotations
  2. Self-label: Run detector on unlabelled manga → candidate annotations
  3. Verify: Send candidates to cloud LLM for verification/correction
     (cheaper than full annotation — just yes/no + bbox corrections)
  4. Retrain: Merge verified annotations into training set, retrain detector
  5. Repeat until convergence or budget exhausted
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from mangaka.schema import MangaPage
from mangaka.utils.bbox import iou

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# Verification prompt (cheaper than full annotation)
# ---------------------------------------------------------------------------

VERIFY_PROMPT = """\
I have a candidate annotation for this manga page. Please verify the accuracy \
of each bounding box and description. For each item, respond with:
- "correct" if the bbox and description are accurate
- "fix" with corrected values if they need adjustment
- "remove" if the detection is a false positive

Candidate annotation:
{annotation_json}

Return ONLY a JSON object:
{{
  "panels": [
    {{
      "index": 0,
      "verdict": "correct" | "fix" | "remove",
      "corrected_bbox": [x1, y1, x2, y2] | null,
      "corrected_description": "..." | null,
      "elements": [
        {{
          "index": 0,
          "verdict": "correct" | "fix" | "remove",
          "corrected_bbox": [x1, y1, x2, y2] | null,
          "corrected_description": "..." | null
        }}
      ]
    }}
  ]
}}"""


# ---------------------------------------------------------------------------
# Distillation round
# ---------------------------------------------------------------------------

async def distill_round(
    detector_path: str,
    unlabelled_dir: str,
    annotation_dir: str,
    provider: str = "claude",
    sample_fraction: float = 0.2,
    max_concurrency: int = 10,
    cfg: dict | None = None,
) -> dict:
    """Run one round of distillation.

    1. Self-label unlabelled images with current detector
    2. Sample a fraction for cloud verification
    3. Apply corrections and save verified annotations

    Returns:
        dict with stats: {self_labelled, verified, corrections, removals}
    """
    cfg = cfg or {}
    unlabelled = Path(unlabelled_dir)
    ann_dir = Path(annotation_dir)

    # Step 1: Self-label with local detector
    console.print("[bold]Step 1:[/bold] Self-labelling with local detector...")
    from mangaka.pipeline.encode import encode_directory

    candidates = encode_directory(
        image_dir=str(unlabelled),
        output_dir=str(ann_dir / "candidates"),
        model_path=detector_path,
        backbone=cfg.get("backbone", "resnet50"),
        gpt2_model=cfg.get("gpt2_model", "gpt2"),
    )
    console.print(f"  Self-labelled {len(candidates)} pages")

    # Step 2: Sample subset for verification
    import random
    sample_size = max(1, int(len(candidates) * sample_fraction))
    to_verify = random.sample(candidates, min(sample_size, len(candidates)))
    console.print(f"  Verifying {len(to_verify)} samples with cloud LLM...")

    # Step 3: Verify with cloud LLM
    from mangaka.annotator import ClaudeAnnotator, OpenAIAnnotator, GeminiAnnotator
    from mangaka.annotator.base import load_image
    from mangaka.annotator.prompt import _extract_json

    annotator_cls = {
        "claude": ClaudeAnnotator,
        "openai": OpenAIAnnotator,
        "gemini": GeminiAnnotator,
    }
    annotator = annotator_cls[provider](max_concurrency=max_concurrency)

    stats = {"self_labelled": len(candidates), "verified": 0, "corrections": 0, "removals": 0}

    sem = asyncio.Semaphore(max_concurrency)

    async def verify_one(json_path: Path) -> Path | None:
        async with sem:
            try:
                page = MangaPage.from_json(json_path)

                # Find the source image
                rel = json_path.relative_to(ann_dir / "candidates")
                img_path = None
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    candidate = unlabelled / rel.with_suffix(ext)
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    return None

                image = load_image(img_path)
                ann_json = page.to_json()
                prompt = VERIFY_PROMPT.format(annotation_json=ann_json)

                raw = await annotator._call_with_retry(image, prompt)
                corrections = _extract_json(raw)

                # Apply corrections
                corrected_page = _apply_corrections(page, corrections)
                if corrected_page is not None:
                    # Save verified annotation
                    verified_path = ann_dir / "verified" / rel
                    verified_path.parent.mkdir(parents=True, exist_ok=True)
                    corrected_page.to_json(verified_path)
                    return verified_path

                return None
            except Exception as e:
                logger.warning("Verification failed for %s: %s", json_path, e)
                return None

    verified_paths = await asyncio.gather(
        *(verify_one(p) for p in to_verify)
    )

    stats["verified"] = sum(1 for p in verified_paths if p is not None)
    console.print(f"  Verified: {stats['verified']} pages")

    return stats


def _apply_corrections(page: MangaPage, corrections: dict) -> MangaPage | None:
    """Apply verification corrections to a MangaPage.

    Returns corrected page, or None if entirely rejected.
    """
    import copy

    corrected = page.model_copy(deep=True)
    panels_to_keep = []

    for panel_corr in corrections.get("panels", []):
        pi = panel_corr.get("index", 0)
        if pi >= len(corrected.panels):
            continue

        panel = corrected.panels[pi]
        verdict = panel_corr.get("verdict", "correct")

        if verdict == "remove":
            continue
        elif verdict == "fix":
            if panel_corr.get("corrected_bbox"):
                bbox = tuple(float(v) for v in panel_corr["corrected_bbox"])
                panel.bbox = bbox
            if panel_corr.get("corrected_description"):
                panel.description = panel_corr["corrected_description"]

        # Process elements
        elems_to_keep = []
        for elem_corr in panel_corr.get("elements", []):
            ei = elem_corr.get("index", 0)
            if ei >= len(panel.elements):
                continue

            elem = panel.elements[ei]
            ev = elem_corr.get("verdict", "correct")

            if ev == "remove":
                continue
            elif ev == "fix":
                if elem_corr.get("corrected_bbox"):
                    bbox = tuple(float(v) for v in elem_corr["corrected_bbox"])
                    elem.bbox = bbox
                if elem_corr.get("corrected_description"):
                    elem.description = elem_corr["corrected_description"]

            elems_to_keep.append(elem)

        panel.elements = elems_to_keep
        panels_to_keep.append(panel)

    if not panels_to_keep:
        return None

    corrected.panels = panels_to_keep
    return corrected


# ---------------------------------------------------------------------------
# Full distillation pipeline
# ---------------------------------------------------------------------------

async def run_distillation(cfg: dict):
    """Run the full iterative distillation pipeline."""
    max_rounds = cfg.get("max_rounds", 5)
    detector_path = cfg["detector_path"]
    unlabelled_dir = cfg["unlabelled_dir"]
    annotation_dir = cfg["annotation_dir"]

    for round_num in range(max_rounds):
        console.print(f"\n[bold cyan]--- Distillation Round {round_num + 1}/{max_rounds} ---[/bold cyan]")

        stats = await distill_round(
            detector_path=detector_path,
            unlabelled_dir=unlabelled_dir,
            annotation_dir=annotation_dir,
            provider=cfg.get("provider", "claude"),
            sample_fraction=cfg.get("verification_sample", 0.2),
            max_concurrency=cfg.get("max_concurrency", 10),
            cfg=cfg,
        )

        console.print(f"  Round stats: {stats}")

        # Retrain detector on merged data (original + verified)
        if stats["verified"] > 0:
            console.print("[bold]Retraining detector...[/bold]")
            _retrain_detector(cfg, annotation_dir)
            detector_path = cfg.get("output_path", detector_path)

        console.print(f"  Round {round_num + 1} complete.")


def _retrain_detector(cfg: dict, annotation_dir: str):
    """Retrain the detector with merged annotations."""
    import subprocess
    import sys

    cmd = [
        sys.executable, "scripts/train_detector.py",
        "--image-dir", cfg["unlabelled_dir"],
        "--annotation-dir", str(Path(annotation_dir) / "verified"),
        "--epochs", str(cfg.get("retrain_epochs", 20)),
        "--lr", str(cfg.get("retrain_lr", 1e-4)),
        "--save-path", cfg.get("output_path", "checkpoints/detector_distilled.safetensors"),
    ]
    subprocess.run(cmd, check=True)
