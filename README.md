# Mangaka

Hierarchical manga analysis and generation pipeline.

## Overview

Mangaka converts manga pages into structured JSON representations and back into
images. The pipeline has four stages:

1. **Cloud Annotation** — Send manga pages to cloud vision LLMs (Claude, GPT-4o,
   Gemini) to get bounding boxes and natural-language descriptions for every
   panel and element.
2. **Train Local Models** — Train a hierarchical detector with a GPT-2
   description head (distilled from the cloud LLM) and a Stable Diffusion
   conditioned infiller.
3. **Distillation Loop** — Iteratively improve the local detector by
   self-labelling manga, verifying with the cloud LLM, and retraining.
4. **Dataset & Generation** — Encode manga into a JSON dataset for RAG / LLM
   finetuning. Generate new manga from JSON using the conditioned infiller.

## Quick Start

```bash
pip install -e .
```

Manga source: **5,574 pages** in `/Volumes/bookofloam/manga_translate/yolo/images/`

### 1. Annotate manga with a cloud LLM

```bash
python scripts/annotate.py \
    --provider claude \
    --image-dir /Volumes/bookofloam/manga_translate/yolo/images \
    --output-dir data/annotations
```

### 2. Train the detector

```bash
torchrun --nproc_per_node=4 scripts/train_detector.py \
    --image-dir /Volumes/bookofloam/manga_translate/yolo/images \
    --annotation-dir data/annotations \
    --config configs/detector.yaml
```

### 3. Train the infiller

```bash
accelerate launch scripts/train_infiller.py \
    --image-dir /Volumes/bookofloam/manga_translate/yolo/images \
    --annotation-dir data/annotations \
    --config configs/infiller.yaml
```

### 4. Build the JSON manga dataset

```bash
python scripts/build_dataset.py \
    --image-dir /Volumes/bookofloam/manga_translate/yolo/images \
    --annotation-dir data/annotations \
    --output-dir data/manga_dataset
```

### 5. Index for RAG

```bash
python scripts/index_dataset.py \
    --dataset-dir data/manga_dataset \
    --index-dir data/manga_index
```

### 6. Generate manga from a plot

```bash
python scripts/generate_manga.py \
    --plot "A samurai confronts his rival in the rain" \
    --index-dir data/manga_index \
    --num-pages 4 \
    --render --infiller-dir checkpoints/infiller/phase2
```

### 7. Export finetuning data

```bash
python scripts/export_finetune.py \
    --dataset-dir data/manga_dataset \
    --output data/finetune/train.jsonl \
    --format chat
```

## JSON Schema

Every manga page is represented as:

```json
{
  "width": 1654,
  "height": 2339,
  "description": "A tense confrontation scene in the rain...",
  "panels": [
    {
      "bbox": [0.02, 0.01, 0.98, 0.45],
      "description": "Wide establishing shot of a rainy cityscape",
      "elements": [
        {
          "bbox": [0.3, 0.1, 0.7, 0.9],
          "type": "character",
          "description": "A young samurai standing in the rain, katana drawn",
          "text": null
        },
        {
          "bbox": [0.05, 0.05, 0.35, 0.25],
          "type": "speech_bubble",
          "description": "Oval speech bubble with clean edges",
          "text": "I won't back down!"
        }
      ]
    }
  ]
}
```

## Project Structure

```
mangaka/
├── mangaka/           # Core library
│   ├── schema.py      # MangaPage / Panel / MangaElement pydantic models
│   ├── annotator/     # Cloud LLM annotation providers
│   ├── detector/      # Hierarchical detector + GPT-2 head
│   ├── infiller/      # SD inpainting + ControlNet layout
│   ├── pipeline/      # Encode / decode / dataset builder
│   └── utils/         # Bbox math, visualization
├── configs/           # YAML config files
└── scripts/           # CLI entry points
```
