"""Modal deployment for mangaka.

Central app definition with:
  - MangaWorker (A10G, encode + decode inference)

Usage:
    modal deploy modal_app.py
    modal run modal_app.py::decode_page --page-json '{"width": 800, ...}'
"""

from __future__ import annotations

import modal

# ---------------------------------------------------------------------------
# App + images + volumes
# ---------------------------------------------------------------------------

app = modal.App("mangaka")

worker_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "torchvision>=0.16",
        "transformers>=4.36",
        "diffusers>=0.25",
        "accelerate>=0.25",
        "safetensors>=0.4",
        "peft>=0.7",
        "pydantic>=2.5",
        "Pillow>=10.0",
        "numpy>=1.24",
        "sentencepiece>=0.1",
    )
    .add_local_python_source("mangaka")
)

ckpt_vol = modal.Volume.from_name("mangaka-checkpoints", create_if_missing=True)
hf_vol = modal.Volume.from_name("mangaka-hf-cache", create_if_missing=True)

CKPT_PATH = "/checkpoints"
HF_HOME = "/hf-cache"


# ---------------------------------------------------------------------------
# MangaWorker (GPU inference for encode + decode)
# ---------------------------------------------------------------------------

@app.cls(
    image=worker_image,
    gpu="A10G",
    volumes={CKPT_PATH: ckpt_vol, HF_HOME: hf_vol},
    scaledown_window=120,
    allow_concurrent_inputs=4,
    timeout=600,
)
class MangaWorker:
    """Stateful inference worker for manga encode/decode.

    Loads detector and infiller models once at container startup,
    serves encode and decode requests.
    The mapper is invoked, never instantiated.
    """

    @modal.enter()
    def load_models(self):
        import os
        import torch
        from pathlib import Path

        os.environ["HF_HOME"] = HF_HOME
        os.environ["TRANSFORMERS_CACHE"] = HF_HOME

        self.device = torch.device("cuda")
        ckpt_dir = Path(CKPT_PATH)

        # -- Load detector ---------------------------------------------------
        detector_path = ckpt_dir / "detector.safetensors"
        self._detector = None
        self._tokenizer = None

        if detector_path.exists():
            from safetensors.torch import load_file as safetensors_load
            from transformers import GPT2Tokenizer
            from mangaka.detector.model import MangaDetectorNet

            print(f"[worker] Loading detector from {detector_path}")
            self._detector = MangaDetectorNet(
                backbone="resnet50", gpt2_model="gpt2",
            ).to(self.device)
            state = safetensors_load(str(detector_path), device=str(self.device))
            self._detector.load_state_dict(state)
            self._detector.eval()
            self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            print("[worker] Detector loaded")
        else:
            print(f"[worker] No detector checkpoint at {detector_path}, encode disabled")

        # -- Load infiller ---------------------------------------------------
        self._infiller = None
        infiller_dir = ckpt_dir / "infiller"

        from mangaka.infiller.model import MangaInfiller, LayoutControlNet

        controlnet = LayoutControlNet()
        controlnet_path = infiller_dir / "controlnet_best.pt"
        if controlnet_path.exists():
            controlnet.load_state_dict(
                torch.load(str(controlnet_path), map_location=self.device)
            )
            print(f"[worker] ControlNet loaded from {controlnet_path}")

        sd_model_id = "stabilityai/stable-diffusion-2-inpainting"
        self._infiller = MangaInfiller(
            sd_model_id=sd_model_id,
            resolution=512,
            controlnet=controlnet,
        )
        self._infiller.load_sd_pipeline(self.device)

        # Auto-detect LoRA weights
        lora_path = infiller_dir / "unet_lora"
        if lora_path.exists() and (lora_path / "adapter_config.json").exists():
            self._infiller.load_lora_weights(lora_path)
            print(f"[worker] LoRA loaded from {lora_path}")

        hf_vol.commit()
        print("[worker] All models loaded, ready for inference")

    @modal.method()
    def decode(
        self,
        page_json: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        style_override: str | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> bytes:
        """Render MangaPage JSON into a PNG image. Returns PNG bytes."""
        import io
        from mangaka.schema import MangaPage

        if self._infiller is None:
            raise RuntimeError("Infiller not loaded")

        page = MangaPage.from_json(page_json)
        image = self._infiller.render_page(
            page,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            style_override=style_override,
            negative_prompt=negative_prompt,
            seed=seed,
        )

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def encode(
        self,
        image_bytes: bytes,
        max_detections: int = 50,
    ) -> str:
        """Detect panels/elements from a PNG image. Returns MangaPage JSON."""
        import io
        from PIL import Image
        from mangaka.pipeline.encode import encode_page

        if self._detector is None or self._tokenizer is None:
            raise RuntimeError("Detector not loaded")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        page = encode_page(
            image, self._detector, self._tokenizer,
            self.device, max_detections=max_detections,
        )
        return page.to_json()
