"""
Bounding-box math: IoU, NMS, coordinate transforms, retina mapping.

All public functions accept plain tuples / tensors of (x1, y1, x2, y2).
Normalised coordinates are in [0, 1]; retina coordinates are in [0, RETINA_SIZE].
"""

from __future__ import annotations

import torch
import numpy as np
from PIL import Image


RETINA_SIZE = 1024


# ---------------------------------------------------------------------------
# Basic geometry (numpy / python)
# ---------------------------------------------------------------------------

def area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Intersection-over-union of two boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = area(a)
    ab = area(b)
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes.  (N, 4) x (M, 4) -> (N, M)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Non-maximum suppression.  Returns indices of kept boxes."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# Coordinate transforms
# ---------------------------------------------------------------------------

def normalise_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Convert pixel coords to normalised [0, 1] coords."""
    x1, y1, x2, y2 = bbox
    return (x1 / width, y1 / height, x2 / width, y2 / height)


def denormalise_bbox(
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert normalised [0, 1] coords to pixel coords."""
    x1, y1, x2, y2 = bbox
    return (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height))


def clip_bbox(
    bbox: tuple[float, float, float, float],
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> tuple[float, float, float, float]:
    """Clip bbox coordinates to [min_val, max_val]."""
    x1, y1, x2, y2 = bbox
    return (
        max(min_val, min(x1, max_val)),
        max(min_val, min(y1, max_val)),
        max(min_val, min(x2, max_val)),
        max(min_val, min(y2, max_val)),
    )


# ---------------------------------------------------------------------------
# Retina mapping  (adapted from nissl)
# ---------------------------------------------------------------------------

def scale_and_pad(
    image: Image.Image,
    pad_color: tuple[int, int, int] = (255, 255, 255),
    retina_size: int = RETINA_SIZE,
) -> tuple[Image.Image, float, int, int]:
    """Scale *image* to fit retina_size x retina_size, centre on canvas.

    Returns (padded_image, scale, offset_x, offset_y).
    """
    w, h = image.size
    if w == 0 or h == 0:
        canvas = Image.new("RGB", (retina_size, retina_size), pad_color)
        return canvas, 1.0, 0, 0
    scale = min(retina_size / w, retina_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (retina_size, retina_size), pad_color)
    ox = (retina_size - new_w) // 2
    oy = (retina_size - new_h) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, scale, ox, oy


def bbox_to_retina(
    bbox: tuple[float, float, float, float],
    scale: float,
    ox: int,
    oy: int,
) -> tuple[int, int, int, int]:
    """Map original-image bbox (pixel) to retina coordinates."""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale + ox),
        int(y1 * scale + oy),
        int(x2 * scale + ox),
        int(y2 * scale + oy),
    )


def retina_to_source(
    retina_bbox: tuple[float, float, float, float],
    scale: float,
    ox: int,
    oy: int,
) -> tuple[int, int, int, int]:
    """Map retina-space bbox back to source-image pixel coordinates."""
    x1, y1, x2, y2 = retina_bbox
    return (
        int((x1 - ox) / scale),
        int((y1 - oy) / scale),
        int((x2 - ox) / scale),
        int((y2 - oy) / scale),
    )


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def bbox_tensor_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between two bbox tensors.  (N, 4) x (M, 4) -> (N, M)."""
    ix1 = torch.max(a[:, None, 0], b[None, :, 0])
    iy1 = torch.max(a[:, None, 1], b[None, :, 1])
    ix2 = torch.min(a[:, None, 2], b[None, :, 2])
    iy2 = torch.min(a[:, None, 3], b[None, :, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return torch.where(union > 0, inter / union, torch.zeros_like(inter))
