"""Evaluation metrics for manga detection and generation quality."""

from mangaka.eval.detection import detection_metrics, match_boxes, per_class_metrics
from mangaka.eval.generation import (
    element_distribution,
    layout_validity_score,
    page_summary,
    structural_similarity,
)

__all__ = [
    "detection_metrics",
    "match_boxes",
    "per_class_metrics",
    "layout_validity_score",
    "structural_similarity",
    "element_distribution",
    "page_summary",
]
