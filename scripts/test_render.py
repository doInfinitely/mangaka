"""Quick test: render a manga page JSON using the trained infiller."""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mangaka.schema import MangaPage
from mangaka.infiller.model import MangaInfiller, LayoutControlNet
from mangaka.pipeline.decode import decode_page

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load ControlNet weights
controlnet = LayoutControlNet()
cn_path = Path("checkpoints/infiller/phase2/controlnet_best.pt")
controlnet.load_state_dict(torch.load(cn_path, map_location=device))
print(f"Loaded ControlNet from {cn_path}")

# Build infiller
infiller = MangaInfiller(
    sd_model_id="runwayml/stable-diffusion-inpainting",
    resolution=512,
    controlnet=controlnet,
)
infiller.load_sd_pipeline(device)

# Load the LLM-generated page
page = MangaPage.from_json("output/test_generation/page_000.json")
print(f"Page: {page.width}x{page.height}, {page.num_panels} panels, {page.num_elements} elements")

# Render with consistent manga style
print("Rendering...")
image = decode_page(
    page, infiller,
    num_inference_steps=30,
    guidance_scale=7.5,
    # style_override, negative_prompt, and seed now default to manga style
)

out_path = Path("output/test_generation/page_000.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
image.save(str(out_path))
print(f"Saved to {out_path}")
