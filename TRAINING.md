# Training the Manga Infiller (LoRA + ControlNet)

## Overview

The infiller renders MangaPage JSONs into images using Stable Diffusion inpainting
with a custom ControlNet layout adapter. Training has two phases:

1. **Phase 1 (epochs 1–20)**: Train ControlNet only, SD UNet frozen
2. **Phase 2 (epochs 21–100)**: Train LoRA adapters on UNet attention layers

## Requirements

- **GPU**: 24GB+ VRAM recommended (A10G, RTX 3090/4090, A100)
- **Python**: 3.11+
- **Dependencies**: `pip install -e ".[train]"` or:
  ```bash
  pip install torch torchvision diffusers transformers accelerate peft \
    safetensors pydantic Pillow numpy sentence-transformers pyyaml
  ```

## Dataset

The trainer expects two directories with matching file structures:

```
images/
  vol01/page001.jpg
  vol01/page002.jpg
  ...
annotations/
  vol01/page001.json   # MangaPage JSON
  vol01/page002.json
  ...
```

Each JSON must be a valid MangaPage (panels with bounding boxes, elements with
types and descriptions). The RAG index already contains 132 annotated pages —
extract them with:

```bash
python -c "
import json
from pathlib import Path
meta = json.load(open('mangaka/rag/index_data/meta.json'))
out = Path('data/training/annotations')
out.mkdir(parents=True, exist_ok=True)
for i, page in enumerate(meta['pages_json']):
    (out / f'page_{i:04d}.json').write_text(json.dumps(page, indent=2))
print(f'Extracted {len(meta[\"pages_json\"])} annotations')
"
```

You will also need the corresponding source images in `data/training/images/`
with matching filenames (e.g., `page_0000.jpg`).

## Training

### Single GPU

```bash
accelerate launch scripts/train_infiller.py \
  --image-dir data/training/images \
  --annotation-dir data/training/annotations \
  --config configs/infiller.yaml
```

### Multi-GPU

```bash
accelerate launch --multi_gpu --num_processes 4 scripts/train_infiller.py \
  --image-dir data/training/images \
  --annotation-dir data/training/annotations \
  --config configs/infiller.yaml
```

### Key config overrides

```bash
# Fewer epochs for quick test
--epochs 10

# Smaller batch size for 16GB GPUs
--batch-size 2

# Custom output directory
--save-path /path/to/output/checkpoints
```

## Config

See `configs/infiller.yaml` for all options. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `freeze_sd_epochs` | 20 | Epochs to train ControlNet only |
| `lora.rank` | 16 | LoRA rank (higher = more capacity, more VRAM) |
| `lora.target_modules` | `[to_k, to_q, to_v, to_out.0]` | UNet attention layers to adapt |
| `batch_size` | 4 | Per-GPU batch size |
| `gradient_accumulation_steps` | 4 | Effective batch = batch_size × accum × num_gpus |

## Output

After training, checkpoints are saved to `checkpoints/infiller/`:

```
checkpoints/infiller/
  controlnet_best.pt     # ControlNet weights
  unet_lora/             # LoRA adapter (PEFT format)
    adapter_config.json
    adapter_model.safetensors
```

## Deploying to Modal

Upload the trained weights to the Modal volume:

```bash
# Install modal CLI
pip install modal

# Upload ControlNet
modal volume put mangaka-checkpoints checkpoints/infiller/controlnet_best.pt infiller/controlnet_best.pt

# Upload LoRA adapter
modal volume put mangaka-checkpoints checkpoints/infiller/unet_lora/ infiller/unet_lora/
```

Then redeploy the Modal app:

```bash
modal deploy modal_app.py
```

The MangaWorker will automatically detect and load the ControlNet + LoRA weights
on startup (see `modal_app.py` lines 108–127).
