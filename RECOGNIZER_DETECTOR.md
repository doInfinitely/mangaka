# Recognizer → Detector Architecture

## Problem

Training a detector requires bbox-annotated data. The 132 GPT-annotated pages have
poor spatial precision. YOLO needs a separate model + bbox training data.

## Pipeline

### Phase 1: DanbooruRecognizer

Train a **recognizer** — "is this crop an X?"

- Architecture: ResNet50 backbone + classification head (no Qwen LM head)
- Training data: Danbooru dataset (millions of tagged images, no bboxes needed)
- Input: an image crop
- Output: type classification + confidence score
- Types: character, speech_bubble, sfx, background, effect, object

### Phase 2: Scan + NMS Detection

Convert the recognizer into a **detector** at inference time — "where are the Xs?"

No training here. Pure geometry:

1. **Hierarchical crop scanning** — slide windows / subdivide manga pages at
   multiple scales (preserving aspect ratio, padding)
2. **Run recognizer** on every candidate crop → type + confidence
3. **Containment-aware NMS** — if box A contains box B and both are confident,
   suppress A. Score: `confidence / sqrt(area)`. Smallest confident crop wins.

Result: tight bounding boxes without any bbox training data.

### Phase 3: Llama Annotation

Generate descriptions for the tight crops:

- Feed each tight crop into Llama (vision model)
- Llama writes a natural-language description of what's in the crop
- Danbooru tags can be fed as context to improve description quality
- Cross-validate recognizer type predictions against tags to filter errors

### Phase 4: Train Qwen Description Head

- Train the Qwen2.5-0.5B LM head on the Llama-described crops
- Input: visual features (ResNet ROI-pooled from tight crop)
- Output: description text matching Llama's annotation
- After training, Qwen replaces Llama at inference time (faster, no API needed)

## Data flow

```
Danbooru (tags) ──→ Phase 1: Train ResNet classifier
                         │
                         ▼
Manga pages ────────→ Phase 2: Scan + NMS ──→ tight crops + types
                                                    │
                                                    ▼
                                           Phase 3: Llama describes crops
                                                    │
                                                    ▼
                                           Phase 4: Train Qwen head
                                                    │
                                                    ▼
                                           Final: ResNet + Qwen = full detector
```

## Key files

| File | Role |
|------|------|
| `mangaka/detector/model.py` | MangaDetectorNet — ResNet backbone + heads |
| `mangaka/pipeline/encode.py` | Current detection loop — to be replaced with scan + NMS |
| `configs/detector.yaml` | Model config |
| `scripts/train_detector.py` | Training script — adapt for Danbooru tag data |

## Notes

- The 132 GPT-annotated pages are **deprecated** — not used in this pipeline
- No self-training loop — recognizer trains once on Danbooru, detection is geometry
- Llama is only used offline for annotation, Qwen serves at inference time
