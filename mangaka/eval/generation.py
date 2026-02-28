"""Generation quality assessment — layout validity and structural comparison."""

from __future__ import annotations

from collections import Counter

import numpy as np

from mangaka.schema import ELEMENT_TYPES, MangaPage
from mangaka.utils.bbox import area, iou


# ---------------------------------------------------------------------------
# Layout validity
# ---------------------------------------------------------------------------


def layout_validity_score(page: MangaPage) -> dict:
    """Check structural validity of a generated MangaPage.

    Returns a dict with individual checks and an overall score in [0, 1].

    Checks:
    - bboxes_in_range: All bboxes have coords in [0, 1]
    - panels_no_excessive_overlap: Panel pairwise IoU < 0.5
    - elements_within_panels: Element bboxes within [0, 1] (panel-relative)
    - has_panels: Page has at least one panel
    """
    checks: dict[str, bool] = {}

    # Check 1: at least one panel
    checks["has_panels"] = len(page.panels) > 0

    # Check 2: all panel bboxes in [0, 1]
    panel_bboxes_ok = True
    for panel in page.panels:
        for v in panel.bbox:
            if v < -0.01 or v > 1.01:
                panel_bboxes_ok = False
                break
    checks["panel_bboxes_in_range"] = panel_bboxes_ok

    # Check 3: panels don't overlap excessively
    panels_ok = True
    for i in range(len(page.panels)):
        for j in range(i + 1, len(page.panels)):
            if iou(page.panels[i].bbox, page.panels[j].bbox) > 0.5:
                panels_ok = False
                break
    checks["panels_no_excessive_overlap"] = panels_ok

    # Check 4: element bboxes within [0, 1] (panel-relative)
    elem_bboxes_ok = True
    for panel in page.panels:
        for elem in panel.elements:
            for v in elem.bbox:
                if v < -0.01 or v > 1.01:
                    elem_bboxes_ok = False
                    break
    checks["element_bboxes_in_range"] = elem_bboxes_ok

    # Overall score: fraction of checks passed
    passed = sum(1 for v in checks.values() if v)
    score = passed / len(checks) if checks else 0.0

    return {
        "checks": checks,
        "score": score,
        "num_passed": passed,
        "num_checks": len(checks),
    }


# ---------------------------------------------------------------------------
# Structural similarity
# ---------------------------------------------------------------------------


def structural_similarity(a: MangaPage, b: MangaPage) -> float:
    """Compare two pages by panel/element counts and spatial layout.

    Returns a similarity score in [0, 1]. Higher means more similar.
    Components:
    - Panel count similarity (Gaussian kernel)
    - Element count similarity (Gaussian kernel)
    - Panel centroid spatial distribution (mean L2 distance)
    """
    scores: list[float] = []

    # Panel count similarity
    panel_diff = abs(a.num_panels - b.num_panels)
    scores.append(np.exp(-0.5 * panel_diff**2))

    # Element count similarity
    elem_diff = abs(a.num_elements - b.num_elements)
    scores.append(np.exp(-0.25 * elem_diff**2))

    # Spatial similarity of panel centroids
    def _centroids(page: MangaPage) -> list[tuple[float, float]]:
        centers = []
        for p in page.panels:
            cx = (p.bbox[0] + p.bbox[2]) / 2
            cy = (p.bbox[1] + p.bbox[3]) / 2
            centers.append((cx, cy))
        return centers

    ca = _centroids(a)
    cb = _centroids(b)

    if ca and cb:
        # Pad the shorter list to match
        max_len = max(len(ca), len(cb))
        while len(ca) < max_len:
            ca.append((0.5, 0.5))
        while len(cb) < max_len:
            cb.append((0.5, 0.5))

        # Sort by position for stable comparison
        ca.sort()
        cb.sort()

        dists = [
            np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
            for (ax, ay), (bx, by) in zip(ca, cb)
        ]
        mean_dist = np.mean(dists)
        spatial_sim = np.exp(-2.0 * mean_dist)
        scores.append(float(spatial_sim))
    elif not ca and not cb:
        scores.append(1.0)
    else:
        scores.append(0.0)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Element distribution
# ---------------------------------------------------------------------------


def element_distribution(page: MangaPage) -> dict:
    """Element type histogram for a page.

    Returns dict with counts and fractions per element type.
    """
    counter: Counter[str] = Counter()
    for panel in page.panels:
        for elem in panel.elements:
            counter[elem.type] += 1

    total = sum(counter.values())
    distribution: dict[str, dict] = {}
    for etype in ELEMENT_TYPES:
        count = counter.get(etype, 0)
        distribution[etype] = {
            "count": count,
            "fraction": count / total if total > 0 else 0.0,
        }

    return {
        "total_elements": total,
        "distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Page summary
# ---------------------------------------------------------------------------


def page_summary(page: MangaPage) -> dict:
    """Combined quality report for a single page."""
    validity = layout_validity_score(page)
    dist = element_distribution(page)

    # Panel area stats
    panel_areas = [area(p.bbox) for p in page.panels]
    coverage = sum(panel_areas)

    return {
        "num_panels": page.num_panels,
        "num_elements": page.num_elements,
        "validity": validity,
        "element_distribution": dist,
        "panel_coverage": float(coverage),
        "mean_panel_area": float(np.mean(panel_areas)) if panel_areas else 0.0,
    }
