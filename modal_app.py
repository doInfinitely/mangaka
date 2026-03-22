"""Modal deployment for mangaka.

Central app definition with:
  - MangaWorker (A10G, encode + decode inference)
  - build_style_bank (A10G, batch-encode artist images into style embeddings)
  - train_style_pipeline (local entrypoint — full style transfer training)

Usage:
    modal deploy modal_app.py
    modal run modal_app.py::decode_page --page-json '{"width": 800, ...}'
    modal run modal_app.py::build_style_bank --image-path /data/images
    modal run modal_app.py::train_style_pipeline --num-artists 200
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
        "requests>=2.31",
        "tqdm>=4.66",
    )
    .add_local_python_source("mangaka")
)

ckpt_vol = modal.Volume.from_name("mangaka-checkpoints", create_if_missing=True)
hf_vol = modal.Volume.from_name("mangaka-hf-cache", create_if_missing=True)
data_vol = modal.Volume.from_name("mangaka-data", create_if_missing=True)

CKPT_PATH = "/checkpoints"
HF_HOME = "/hf-cache"
DATA_PATH = "/data"

# Lightweight CPU image for gallery fetching (no torch/GPU deps)
fetcher_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "requests>=2.31",
        "Pillow>=10.0",
        "pydantic>=2.5",
        "tqdm>=4.66",
    )
    .add_local_python_source("mangaka")
)


# ---------------------------------------------------------------------------
# MangaWorker (GPU inference for encode + decode)
# ---------------------------------------------------------------------------

@app.cls(
    image=worker_image,
    gpu="A10G",
    volumes={CKPT_PATH: ckpt_vol, HF_HOME: hf_vol, DATA_PATH: data_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    scaledown_window=120,
    timeout=600,
)
@modal.concurrent(max_inputs=1)
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
            from transformers import AutoTokenizer
            from mangaka.detector.model import MangaDetectorNet

            print(f"[worker] Loading detector from {detector_path}")
            self._detector = MangaDetectorNet(
                backbone="resnet50", lm_model="Qwen/Qwen2.5-0.5B",
            ).to(self.device)
            state = safetensors_load(str(detector_path), device=str(self.device))
            self._detector.load_state_dict(state)
            self._detector.eval()
            self._tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
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

        sd_model_id = "runwayml/stable-diffusion-inpainting"
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

        # -- Load style applicator (lazy — only loads weights on first stylize call)
        # Create whenever generator checkpoint exists. Bank dir may not exist
        # yet at startup — it gets created on-demand by prepare_artist_style.
        self._style_applicator = None
        style_gen_path = ckpt_dir / "style" / "generator_best.pt"
        self._style_bank_dir = Path(DATA_PATH) / "style_bank"
        if style_gen_path.exists():
            from mangaka.style.apply import StyleApplicator

            self._style_applicator = StyleApplicator(
                bank_dir=str(self._style_bank_dir),
                generator_path=str(style_gen_path),
                device="cuda",
            )
            print(f"[worker] StyleApplicator configured (bank={self._style_bank_dir})")
        else:
            print(f"[worker] No style generator at {style_gen_path}, style transfer disabled")

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
        artist_names: list[str] | None = None,
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

        # Apply style transfer if artists provided and applicator available
        if artist_names and self._style_applicator is not None:
            # Reload volume + clear cache to see banks written by prepare_artist_style
            data_vol.reload()
            self._style_applicator.clear_bank_cache()
            image = self._style_applicator.stylize(image, artist_names)

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


# ---------------------------------------------------------------------------
# Style bank builder (batch GPU job)
# ---------------------------------------------------------------------------

@app.function(
    image=worker_image,
    gpu="A10G",
    volumes={DATA_PATH: data_vol},
    timeout=14400,
)
def build_style_bank(
    image_path: str = "/data/images",
    bank_path: str = "/data/style_bank",
    resolution: int = 512,
    batch_size: int = 64,
    style_dim: int = 512,
):
    """Encode all artist images into style embeddings on GPU.

    Expects image_path to contain artist subdirectories:
        /data/images/artist_001/*.png
        /data/images/artist_002/*.png

    Writes one .pt file per artist to bank_path:
        /data/style_bank/artist_001.pt  # (N, 512)

    Usage:
        modal run modal_app.py::build_style_bank
        modal run modal_app.py::build_style_bank --image-path /data/images --batch-size 128
    """
    # Reload volume to see images written by fetch stage
    data_vol.reload()

    from pathlib import Path

    import torch
    from PIL import Image
    from torchvision import transforms

    from mangaka.style.encoder import StyleEncoder
    from mangaka.style.bank import StyleBank

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

    device = torch.device("cuda")
    encoder = StyleEncoder(style_dim=style_dim).to(device)
    encoder.eval()
    bank = StyleBank(bank_path, style_dim=style_dim)

    transform = transforms.Compose([
        transforms.Resize(
            (resolution, resolution),
            interpolation=transforms.InterpolationMode.LANCZOS,
        ),
        transforms.ToTensor(),
    ])

    image_dir = Path(image_path)
    artist_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())
    print(f"[style-bank] Found {len(artist_dirs)} artist directories")

    bank_dir = Path(bank_path)
    total_images = 0
    skipped = 0
    for ai, artist_dir in enumerate(artist_dirs):
        artist_id = artist_dir.name

        # Skip artists already encoded
        if (bank_dir / f"{artist_id}.pt").exists():
            skipped += 1
            continue

        images = sorted(
            p for p in artist_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue

        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_paths = images[i : i + batch_size]
            batch_tensors = []
            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_tensors.append(transform(img))
                except Exception as e:
                    print(f"[style-bank] Failed to load {img_path}: {e}")
                    batch_tensors.append(torch.zeros(3, resolution, resolution))

            batch = torch.stack(batch_tensors).to(device)
            with torch.no_grad(), torch.autocast("cuda"):
                embeddings = encoder(batch)
            all_embeddings.append(embeddings.cpu())

        artist_embeddings = torch.cat(all_embeddings, dim=0)
        bank.save_artist(artist_id, artist_embeddings)
        total_images += len(images)

        if (ai + 1) % 10 == 0:
            data_vol.commit()  # periodic commit to avoid losing progress
            print(f"[style-bank] Processed {ai + 1}/{len(artist_dirs)} artists ({total_images} images)")

    data_vol.commit()
    print(f"[style-bank] Done: {len(artist_dirs)} artists, {total_images} images (skipped {skipped} already encoded)")


# ---------------------------------------------------------------------------
# On-demand style preparation (gallery fetch + GPU encode)
# ---------------------------------------------------------------------------

@app.function(
    image=worker_image,
    gpu="A10G",
    volumes={DATA_PATH: data_vol},
    secrets=[modal.Secret.from_name("exhentai-cookies")],
    timeout=1800,
)
def prepare_artist_style(
    artist_names: list[str],
    resolution: int = 512,
    batch_size: int = 64,
    style_dim: int = 512,
    max_galleries_per_artist: int = 20,
) -> dict[str, dict[str, int]]:
    """Fetch galleries + encode style embeddings on GPU for requested artists.

    Runs the full pipeline on Modal: gallery fetching (IO-bound) then
    VGG19 style encoding (GPU). Saves .pt banks to the data volume.

    Called remotely by the Railway /prepare-style endpoint.

    Returns:
        Per-artist status: { "artist_name": { "images": N, "embeddings": N } }
    """
    from mangaka.style.prepare import prepare_artist_styles

    status = prepare_artist_styles(
        artist_names,
        cache_dir=f"{DATA_PATH}/gallery_cache",
        image_dir=f"{DATA_PATH}/images",
        bank_dir=f"{DATA_PATH}/style_bank",
        style_dim=style_dim,
        device="cuda",
        resolution=resolution,
        batch_size=batch_size,
        max_galleries_per_artist=max_galleries_per_artist,
    )

    data_vol.commit()
    return status


# ---------------------------------------------------------------------------
# Style transfer training pipeline (Modal)
# ---------------------------------------------------------------------------

@app.function(
    image=fetcher_image,
    volumes={DATA_PATH: data_vol},
    timeout=120,
)
def upload_artist_index(index_json: str):
    """Write artist index JSON to the data volume."""
    from pathlib import Path

    path = Path(DATA_PATH) / "artist_index.json"
    path.write_text(index_json, encoding="utf-8")
    data_vol.commit()
    print(f"[upload] Wrote artist index to {path}")


@app.function(
    image=fetcher_image,
    volumes={DATA_PATH: data_vol},
    secrets=[modal.Secret.from_name("exhentai-cookies")],
    timeout=3600,
    max_containers=50,
    retries=modal.Retries(
        max_retries=2,
        backoff_coefficient=2.0,
        initial_delay=30.0,
    ),
)
def fetch_artist_batch(
    artist_name: str,
    gallery_urls: list[str],
) -> dict:
    """Fetch all galleries for a single artist. Parallelized via .map().

    Each container gets its own IP, so ExHentai rate-limits are per-container.
    """
    from mangaka.gallery.client import ExHentaiClient
    from mangaka.gallery.cache import GalleryCache
    from mangaka.gallery.fetch_indexed import fetch_artist_from_index

    try:
        client = ExHentaiClient()  # reads EXHENTAI_COOKIES from env
        cache = GalleryCache(cache_dir=f"{DATA_PATH}/gallery_cache")

        paths = fetch_artist_from_index(
            artist_name=artist_name,
            gallery_urls=gallery_urls,
            client=client,
            cache=cache,
            output_dir=f"{DATA_PATH}/images",
        )

        data_vol.commit()
        return {"artist": artist_name, "images": len(paths), "error": None}
    except Exception as exc:
        return {"artist": artist_name, "images": 0, "error": str(exc)}


@app.function(
    image=worker_image,
    gpu="A10G",
    volumes={DATA_PATH: data_vol, CKPT_PATH: ckpt_vol, HF_HOME: hf_vol},
    timeout=86400,
)
def train_style_model(config_overrides: dict | None = None):
    """Train the style transfer GAN on GPU.

    Reads images from /data/images and style banks from /data/style_bank.
    Saves checkpoints to /checkpoints/style/.
    """
    from mangaka.style.train import train_style_gan

    data_vol.reload()

    # Load base config
    cfg = {
        "style_dim": 512,
        "resolution": 512,
        "generator": {"base_channels": 64, "num_heads": 8},
        "discriminator": {"base_channels": 64, "n_layers": 3},
        "max_bank_size": 256,
        "losses": {
            "lambda_l1": 20.0,
            "lambda_perceptual": 5.0,
            "lambda_style": 10.0,
            "lambda_adversarial": 1.0,
        },
        "epochs": 100,
        "batch_size": 4,
        "generator_lr": 1e-4,
        "discriminator_lr": 4e-4,
        "weight_decay": 0.01,
        "val_split": 0.1,
        "num_workers": 4,
        "amp": True,
        "gradient_accumulation_steps": 1,
        "seed": 42,
        "save_path": f"{CKPT_PATH}/style/",
    }

    # Apply overrides from CLI
    if config_overrides:
        for key, value in config_overrides.items():
            if isinstance(value, dict) and key in cfg and isinstance(cfg[key], dict):
                cfg[key].update(value)
            else:
                cfg[key] = value

    print(f"[train] Config: epochs={cfg['epochs']}, batch_size={cfg['batch_size']}, "
          f"amp={cfg['amp']}, losses={cfg['losses']}")
    train_style_gan(
        image_dir=f"{DATA_PATH}/images",
        bank_dir=f"{DATA_PATH}/style_bank",
        cfg=cfg,
    )

    ckpt_vol.commit()
    print("[train] Checkpoints committed to volume")


@app.local_entrypoint()
def train_style_pipeline(
    num_artists: int = 200,
    min_galleries: int = 5,
    max_galleries_per_artist: int = 20,
    pickle_path: str = "/Volumes/trigger/manga-translate/artists.p",
    epochs: int = 100,
    batch_size: int = 4,
    skip_fetch: bool = False,
    skip_encode: bool = False,
    lambda_l1: float = 20.0,
    lambda_perceptual: float = 5.0,
    lambda_style: float = 10.0,
    lambda_adversarial: float = 1.0,
):
    """Full style transfer training pipeline: fetch → encode → train.

    Usage:
        modal run modal_app.py::train_style_pipeline --num-artists 200
        modal run modal_app.py::train_style_pipeline --num-artists 3 --epochs 2  # smoke test
        modal run modal_app.py::train_style_pipeline --skip-fetch --skip-encode --epochs 50  # retrain only
        modal run modal_app.py::train_style_pipeline --lambda-style 5.0 --lambda-l1 30.0  # tune losses
    """
    import json

    from mangaka.gallery.index import load_artist_index, select_top_artists

    # Stage 1 (local): Load and select artists
    print(f"\n=== Stage 1: Loading artist index from {pickle_path} ===")
    full_index = load_artist_index(pickle_path)
    print(f"  Full index: {len(full_index)} artists")

    selected = select_top_artists(
        full_index,
        num_artists=num_artists,
        min_galleries=min_galleries,
        max_galleries_per_artist=max_galleries_per_artist,
    )
    print(f"  Selected: {len(selected)} artists")
    total_galleries = sum(len(urls) for urls in selected.values())
    print(f"  Total galleries to fetch: {total_galleries}")

    if not skip_fetch:
        # Stage 2 (remote, 1 CPU): Upload index
        print("\n=== Stage 2: Uploading artist index to volume ===")
        index_json = json.dumps(selected, ensure_ascii=False)
        upload_artist_index.remote(index_json)
        print("  Index uploaded")

        # Stage 3 (remote, N CPU): Parallel gallery fetching
        print(f"\n=== Stage 3: Fetching galleries ({len(selected)} artists, up to 50 containers) ===")
        names = list(selected.keys())
        url_lists = list(selected.values())

        results = list(fetch_artist_batch.map(names, url_lists))

        succeeded = [r for r in results if r["error"] is None]
        failed = [r for r in results if r["error"] is not None]
        total_images = sum(r["images"] for r in succeeded)
        print(f"  Fetched: {len(succeeded)}/{len(results)} artists, {total_images} images")
        if failed:
            print(f"  Failed ({len(failed)}):")
            for r in failed[:10]:
                print(f"    - {r['artist']}: {r['error']}")
            if len(failed) > 10:
                print(f"    ... and {len(failed) - 10} more")
    else:
        print("\n=== Stages 2-3: Skipped (--skip-fetch) ===")

    if not skip_encode:
        # Stage 4 (remote, 1 GPU): Build style bank
        print("\n=== Stage 4: Building style bank (VGG19 encoding) ===")
        build_style_bank.remote()
        print("  Style bank built")
    else:
        print("\n=== Stage 4: Skipped (--skip-encode) ===")

    # Stage 5 (remote, 1 GPU): Train GAN
    print(f"\n=== Stage 5: Training style GAN (epochs={epochs}, batch_size={batch_size}) ===")
    config_overrides = {
        "epochs": epochs,
        "batch_size": batch_size,
        "losses": {
            "lambda_l1": lambda_l1,
            "lambda_perceptual": lambda_perceptual,
            "lambda_style": lambda_style,
            "lambda_adversarial": lambda_adversarial,
        },
    }
    train_style_model.remote(config_overrides=config_overrides)
    print("  Training complete!")

    print("\n=== Pipeline finished ===")
    print("Verify results:")
    print("  modal volume ls mangaka-data images/")
    print("  modal volume ls mangaka-data style_bank/")
    print("  modal volume ls mangaka-checkpoints style/")
