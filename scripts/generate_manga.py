"""
CLI: Generate manga JSON from a plot description using RAG.

Usage:
    python scripts/generate_manga.py \
        --plot "A samurai confronts his rival in the rain" \
        --index-dir data/manga_index \
        --output-dir output/generated

    # Multi-page:
    python scripts/generate_manga.py \
        --plot "A young ninja discovers her hidden power" \
        --num-pages 4 \
        --output-dir output/generated

    # With rendering:
    python scripts/generate_manga.py \
        --plot "..." \
        --render \
        --infiller-dir checkpoints/infiller/phase2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel as RichPanel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.generate_manga")
console = Console()


async def run(args):
    from mangaka.rag.index import MangaIndex
    from mangaka.rag.generate import generate_manga

    # Load index
    console.print(f"Loading index from [cyan]{args.index_dir}[/cyan]...")
    index = MangaIndex()
    index.load(args.index_dir, embed_model=args.embed_model)
    console.print(f"  Index loaded: {len(index.pages)} pages\n")

    # Build page spec
    page_spec = {}
    if args.panel_count:
        page_spec["panel_count"] = args.panel_count
    if args.width and args.height:
        page_spec["width"] = args.width
        page_spec["height"] = args.height
    if args.style:
        page_spec["style"] = args.style

    structure_hint = {}
    if args.panel_count:
        structure_hint["panel_count"] = args.panel_count

    # Generate
    console.print(
        RichPanel(
            f"[bold]Plot:[/bold] {args.plot}",
            title="Generating manga",
            border_style="blue",
        )
    )

    pages = await generate_manga(
        plot=args.plot,
        index=index,
        provider=args.provider,
        num_pages=args.num_pages,
        top_k=args.top_k,
        page_spec=page_spec or None,
        structure_hint=structure_hint or None,
    )

    if not pages:
        console.print("[red]No pages generated.[/red]")
        return

    # Save JSON output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, page in enumerate(pages):
        json_path = output_dir / f"page_{i:03d}.json"
        page.to_json(json_path)
        console.print(
            f"  Page {i}: {page.num_panels} panels, "
            f"{page.num_elements} elements → [green]{json_path}[/green]"
        )

    # Optionally render with the infiller
    if args.render and args.infiller_dir:
        console.print("\n[bold]Rendering with infiller...[/bold]")
        from mangaka.pipeline.decode import decode_page
        from mangaka.infiller.model import MangaInfiller, LayoutControlNet
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        controlnet = LayoutControlNet()
        cn_path = Path(args.infiller_dir) / "controlnet_best.pt"
        if cn_path.exists():
            controlnet.load_state_dict(torch.load(cn_path, map_location=device))

        infiller = MangaInfiller(
            sd_model_id=args.sd_model,
            resolution=512,
            controlnet=controlnet,
        )
        infiller.load_sd_pipeline(device)

        for i, page in enumerate(pages):
            img = decode_page(page, infiller)
            img_path = output_dir / f"page_{i:03d}.png"
            img.save(str(img_path))
            console.print(f"  Rendered → [green]{img_path}[/green]")

    console.print(f"\n[green]Done![/green] Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate manga from plot description")
    parser.add_argument("--plot", required=True,
                        help="Plot / story description")
    parser.add_argument("--index-dir", default="data/manga_index",
                        help="Path to the manga RAG index")
    parser.add_argument("--output-dir", default="output/generated",
                        help="Directory to save generated JSONs (and images)")
    parser.add_argument("--provider", choices=["claude", "openai", "gemini"],
                        default="claude")
    parser.add_argument("--num-pages", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of examples to retrieve")
    parser.add_argument("--panel-count", type=int, default=None,
                        help="Target panel count per page")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--style", type=str, default=None,
                        help="Art style hint")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2")

    # Rendering options
    parser.add_argument("--render", action="store_true",
                        help="Also render the generated pages with the infiller")
    parser.add_argument("--infiller-dir", default=None,
                        help="Path to infiller checkpoints (required if --render)")
    parser.add_argument("--sd-model",
                        default="stabilityai/stable-diffusion-2-inpainting")
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
