"""Detection quality metrics — precision, recall, F1, mAP.

Compares predicted MangaPage annotations against ground truth using
bounding-box IoU matching.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from mangaka.schema import ELEMENT_TYPES, MangaPage
from mangaka.utils.bbox import iou


# ---------------------------------------------------------------------------
# Box matching
# ---------------------------------------------------------------------------


def match_boxes(
    pred_boxes: list[tuple[float, float, float, float]],
    gt_boxes: list[tuple[float, float, float, float]],
    iou_threshold: float = 0.5,
) -> list[tuple[int, int, float]]:
    """Match predicted boxes to ground-truth boxes using greedy IoU matching.

    Returns a list of (pred_idx, gt_idx, iou_score) for matched pairs.
    Each predicted and ground-truth box is matched at most once.
    """
    if not pred_boxes or not gt_boxes:
        return []

    # Compute all pairwise IoU scores
    pairs: list[tuple[float, int, int]] = []
    for pi, pb in enumerate(pred_boxes):
        for gi, gb in enumerate(gt_boxes):
            score = iou(pb, gb)
            if score >= iou_threshold:
                pairs.append((score, pi, gi))

    # Greedy matching: highest IoU first
    pairs.sort(reverse=True)
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for score, pi, gi in pairs:
        if pi not in matched_pred and gi not in matched_gt:
            matches.append((pi, gi, score))
            matched_pred.add(pi)
            matched_gt.add(gi)

    return matches


# ---------------------------------------------------------------------------
# Aggregate detection metrics
# ---------------------------------------------------------------------------


def _extract_boxes_and_types(
    pages: list[MangaPage],
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Flatten all element boxes and types from a list of pages.

    Element bboxes are converted to page-absolute coordinates.
    """
    results: list[tuple[tuple[float, float, float, float], str]] = []
    for page in pages:
        for panel in page.panels:
            px1, py1, px2, py2 = panel.bbox
            pw = px2 - px1
            ph = py2 - py1
            for elem in panel.elements:
                ex1 = px1 + elem.bbox[0] * pw
                ey1 = py1 + elem.bbox[1] * ph
                ex2 = px1 + elem.bbox[2] * pw
                ey2 = py1 + elem.bbox[3] * ph
                results.append(((ex1, ey1, ex2, ey2), elem.type))
    return results


def detection_metrics(
    predictions: list[MangaPage],
    ground_truth: list[MangaPage],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """Compute precision, recall, F1, and mAP across IoU thresholds.

    Args:
        predictions: Predicted MangaPages (one per image).
        ground_truth: Ground-truth MangaPages (aligned 1:1 with predictions).
        iou_thresholds: IoU thresholds to evaluate at (default [0.5, 0.75]).

    Returns:
        Dictionary with per-threshold and mean metrics.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    results: dict = {"thresholds": {}}

    for thresh in iou_thresholds:
        total_tp = 0
        total_pred = 0
        total_gt = 0

        for pred_page, gt_page in zip(predictions, ground_truth):
            pred_items = _extract_boxes_and_types([pred_page])
            gt_items = _extract_boxes_and_types([gt_page])

            pred_boxes = [b for b, _ in pred_items]
            gt_boxes = [b for b, _ in gt_items]

            matches = match_boxes(pred_boxes, gt_boxes, iou_threshold=thresh)
            total_tp += len(matches)
            total_pred += len(pred_boxes)
            total_gt += len(gt_boxes)

        precision = total_tp / total_pred if total_pred > 0 else float("nan")
        recall = total_tp / total_gt if total_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 and not np.isnan(precision)
            else 0.0
        )

        results["thresholds"][str(thresh)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": total_tp,
            "num_predictions": total_pred,
            "num_ground_truth": total_gt,
        }

    # Mean across thresholds (mAP-style)
    f1_scores = [
        v["f1"]
        for v in results["thresholds"].values()
        if not np.isnan(v.get("f1", float("nan")))
    ]
    results["mean_f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0
    return results


# ---------------------------------------------------------------------------
# Per-class metrics
# ---------------------------------------------------------------------------


def per_class_metrics(
    predictions: list[MangaPage],
    ground_truth: list[MangaPage],
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """Per-element-type precision, recall, and F1.

    Returns a dict keyed by element type with sub-dicts of metrics.
    """
    # Gather per-class boxes
    pred_by_class: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    gt_by_class: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)

    for pred_page, gt_page in zip(predictions, ground_truth):
        for bbox, etype in _extract_boxes_and_types([pred_page]):
            pred_by_class[etype].append(bbox)
        for bbox, etype in _extract_boxes_and_types([gt_page]):
            gt_by_class[etype].append(bbox)

    results: dict[str, dict] = {}
    for etype in ELEMENT_TYPES:
        pred_boxes = pred_by_class.get(etype, [])
        gt_boxes = gt_by_class.get(etype, [])

        matches = match_boxes(pred_boxes, gt_boxes, iou_threshold=iou_threshold)
        tp = len(matches)
        n_pred = len(pred_boxes)
        n_gt = len(gt_boxes)

        precision = tp / n_pred if n_pred > 0 else float("nan")
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 and not np.isnan(precision)
            else 0.0
        )

        results[etype] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "num_predictions": n_pred,
            "num_ground_truth": n_gt,
        }

    return results
