"""
CLI: Annotate manga pages with a cloud vision LLM.

Usage:
    python scripts/annotate.py --provider claude --image-dir /path/to/manga --output-dir /path/to/annotations
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from mangaka.annotator.base import load_image
from mangaka.schema import MangaPage

console = Console()
logger = logging.getLogger("mangaka.annotate")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def build_annotator(args):
    """Instantiate the chosen annotator provider."""
    kwargs = dict(
        max_concurrency=args.max_concurrency,
        retry_attempts=args.retry_attempts,
        retry_delay=args.retry_delay,
        two_pass=args.two_pass,
        crop_padding=args.crop_padding,
    )

    if args.provider == "claude":
        from mangaka.annotator.claude import ClaudeAnnotator
        return ClaudeAnnotator(model=args.model or "claude-sonnet-4-20250514", **kwargs)
    elif args.provider == "openai":
        from mangaka.annotator.openai import OpenAIAnnotator
        return OpenAIAnnotator(model=args.model or "gpt-4o", **kwargs)
    elif args.provider == "gemini":
        from mangaka.annotator.gemini import GeminiAnnotator
        return GeminiAnnotator(model=args.model or "gemini-2.0-flash", **kwargs)
    else:
        raise ValueError(f"Unknown provider: {args.provider}")


def discover_images(image_dir: Path) -> list[Path]:
    """Find all image files in a directory (recursive)."""
    images = []
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file():
            images.append(p)
    return images


async def run(args):
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(image_dir)
    if not image_paths:
        console.print(f"[red]No images found in {image_dir}[/red]")
        sys.exit(1)

    # Skip already-annotated images
    if not args.force:
        remaining = []
        for p in image_paths:
            rel = p.relative_to(image_dir)
            json_path = output_dir / rel.with_suffix(".json")
            if not json_path.exists():
                remaining.append(p)
        skipped = len(image_paths) - len(remaining)
        if skipped > 0:
            console.print(f"Skipping {skipped} already-annotated images")
        image_paths = remaining

    console.print(
        f"Annotating [bold]{len(image_paths)}[/bold] images "
        f"with [cyan]{args.provider}[/cyan]"
    )

    annotator = build_annotator(args)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Annotating...", total=len(image_paths))

        sem = asyncio.Semaphore(args.max_concurrency)

        async def process_one(img_path: Path):
            async with sem:
                try:
                    image = load_image(img_path)
                    w, h = image.size
                    page = await annotator.annotate(image, w, h)

                    rel = img_path.relative_to(image_dir)
                    json_path = output_dir / rel.with_suffix(".json")
                    json_path.parent.mkdir(parents=True, exist_ok=True)
                    page.to_json(json_path)

                    progress.update(task, advance=1)
                    return img_path, page
                except Exception as e:
                    logger.error("Failed to annotate %s: %s", img_path, e)
                    progress.update(task, advance=1)
                    return img_path, None

        results = await asyncio.gather(
            *(process_one(p) for p in image_paths)
        )

    # Summary
    successes = sum(1 for _, page in results if page is not None)
    failures = len(results) - successes
    total_panels = sum(
        page.num_panels for _, page in results if page is not None
    )
    total_elements = sum(
        page.num_elements for _, page in results if page is not None
    )

    console.print(f"\n[green]Done![/green] {successes} annotated, {failures} failed")
    console.print(f"Total: {total_panels} panels, {total_elements} elements")
    console.print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Annotate manga pages with cloud VLM")
    parser.add_argument("--provider", choices=["claude", "openai", "gemini"],
                        default="claude")
    parser.add_argument("--model", type=str, default=None,
                        help="Override model name for the provider")
    parser.add_argument("--image-dir", required=True,
                        help="Directory containing manga images")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save annotation JSONs")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--retry-attempts", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--two-pass", action="store_true", default=True)
    parser.add_argument("--no-two-pass", action="store_false", dest="two_pass")
    parser.add_argument("--crop-padding", type=float, default=0.05)
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if JSON already exists")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
