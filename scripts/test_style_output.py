"""Quick test: render a page with and without style transfer via Modal.

Produces 4 images in output/style_test/:
  - baseline.png        (no style transfer)
  - style_pig_king.png
  - style_crazy_dad.png
  - style_crimson.png

Usage:
    python scripts/test_style_output.py
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

# Minimal 3-panel test page
TEST_PAGE = {
    "width": 800,
    "height": 1200,
    "description": "A warrior faces a dragon in a dark forest",
    "style": "manga, detailed ink drawing, high contrast",
    "panels": [
        {
            "bbox": [0.05, 0.02, 0.95, 0.35],
            "description": "Wide establishing shot of a dark misty forest with towering ancient trees",
            "elements": [
                {
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "type": "background",
                    "description": "Dark misty forest with towering ancient trees, moonlight filtering through canopy",
                },
                {
                    "bbox": [0.3, 0.4, 0.5, 1.0],
                    "type": "character",
                    "description": "Armored warrior standing with sword drawn, seen from behind",
                },
            ],
        },
        {
            "bbox": [0.05, 0.37, 0.50, 0.68],
            "description": "Close-up of the warrior's determined face",
            "elements": [
                {
                    "bbox": [0.1, 0.05, 0.9, 0.95],
                    "type": "character",
                    "description": "Close-up of warrior's face, intense eyes, sweat on brow",
                },
                {
                    "bbox": [0.15, 0.02, 0.85, 0.25],
                    "type": "speech_bubble",
                    "description": "Speech bubble with text",
                    "text": "I won't back down!",
                },
            ],
        },
        {
            "bbox": [0.52, 0.37, 0.95, 0.98],
            "description": "Dragon emerges from the shadows, massive and terrifying",
            "elements": [
                {
                    "bbox": [0.0, 0.0, 1.0, 1.0],
                    "type": "background",
                    "description": "Dark shadows with glowing embers and smoke",
                },
                {
                    "bbox": [0.1, 0.1, 0.95, 0.9],
                    "type": "character",
                    "description": "Massive dragon emerging from shadows, glowing red eyes, scales glistening",
                },
                {
                    "bbox": [0.05, 0.8, 0.6, 0.98],
                    "type": "sfx",
                    "description": "Rumbling sound effect",
                    "text": "GRRRAAAA",
                },
            ],
        },
    ],
}

ARTISTS = ["pig_king", "crazy_dad", "crimson"]


def main():
    output_dir = Path("output/style_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    worker_cls = modal.Cls.from_name("mangaka", "MangaWorker")
    worker = worker_cls()

    page_json = json.dumps(TEST_PAGE)

    # Baseline (no style transfer)
    print("Rendering baseline (no style)...")
    png_bytes = worker.decode.remote(
        page_json,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42,
    )
    baseline_path = output_dir / "baseline.png"
    baseline_path.write_bytes(png_bytes)
    print(f"  -> {baseline_path} ({len(png_bytes)} bytes)")

    # Per-artist style transfer
    for artist in ARTISTS:
        print(f"Rendering with style: {artist}...")
        png_bytes = worker.decode.remote(
            page_json,
            num_inference_steps=30,
            guidance_scale=7.5,
            seed=42,
            artist_names=[artist],
        )
        out_path = output_dir / f"style_{artist}.png"
        out_path.write_bytes(png_bytes)
        print(f"  -> {out_path} ({len(png_bytes)} bytes)")

    print(f"\nDone! Check {output_dir}/")


if __name__ == "__main__":
    main()
