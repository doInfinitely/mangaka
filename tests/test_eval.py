"""Tests for mangaka.eval — detection and generation quality metrics."""

from __future__ import annotations

import math

import pytest

from mangaka.eval.detection import detection_metrics, match_boxes, per_class_metrics
from mangaka.eval.generation import (
    element_distribution,
    layout_validity_score,
    page_summary,
    structural_similarity,
)
from mangaka.schema import MangaElement, MangaPage, Panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page(
    panels: list[tuple[tuple[float, ...], list[tuple[tuple[float, ...], str]]]]
) -> MangaPage:
    """Shorthand to build a MangaPage from panel specs.

    Each panel is (panel_bbox, [(elem_bbox, elem_type), ...]).
    """
    page_panels = []
    for panel_bbox, elements in panels:
        elems = [
            MangaElement(bbox=eb, type=et, description=f"a {et}")
            for eb, et in elements
        ]
        page_panels.append(Panel(bbox=panel_bbox, description="panel", elements=elems))
    return MangaPage(width=1000, height=1000, description="page", panels=page_panels)


# ---------------------------------------------------------------------------
# match_boxes()
# ---------------------------------------------------------------------------


class TestMatchBoxes:
    def test_perfect_match(self):
        boxes = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.5, 1.0, 1.0)]
        matches = match_boxes(boxes, boxes, iou_threshold=0.5)
        assert len(matches) == 2

    def test_no_overlap(self):
        pred = [(0.0, 0.0, 0.3, 0.3)]
        gt = [(0.7, 0.7, 1.0, 1.0)]
        matches = match_boxes(pred, gt, iou_threshold=0.5)
        assert len(matches) == 0

    def test_empty_predictions(self):
        matches = match_boxes([], [(0.0, 0.0, 1.0, 1.0)], iou_threshold=0.5)
        assert len(matches) == 0

    def test_empty_ground_truth(self):
        matches = match_boxes([(0.0, 0.0, 1.0, 1.0)], [], iou_threshold=0.5)
        assert len(matches) == 0


# ---------------------------------------------------------------------------
# detection_metrics()
# ---------------------------------------------------------------------------


class TestDetectionMetrics:
    def test_perfect_detection(self):
        page = _make_page([
            ((0.0, 0.0, 0.5, 0.5), [((0.1, 0.1, 0.9, 0.9), "character")]),
        ])
        result = detection_metrics([page], [page], iou_thresholds=[0.5])
        m = result["thresholds"]["0.5"]
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_no_predictions(self):
        gt = _make_page([
            ((0.0, 0.0, 0.5, 0.5), [((0.1, 0.1, 0.9, 0.9), "character")]),
        ])
        pred = _make_page([])  # no panels = no elements
        result = detection_metrics([pred], [gt], iou_thresholds=[0.5])
        m = result["thresholds"]["0.5"]
        assert math.isnan(m["precision"])  # 0/0
        assert m["recall"] == pytest.approx(0.0)

    def test_partial_overlap_different_thresholds(self):
        # Pred box shifted slightly from GT
        gt = _make_page([
            ((0.0, 0.0, 1.0, 1.0), [((0.0, 0.0, 0.5, 0.5), "character")]),
        ])
        pred = _make_page([
            ((0.0, 0.0, 1.0, 1.0), [((0.1, 0.1, 0.6, 0.6), "character")]),
        ])
        result = detection_metrics([pred], [gt], iou_thresholds=[0.3, 0.9])
        # At IoU 0.3 should match; at 0.9 should not
        assert result["thresholds"]["0.3"]["tp"] >= 1
        assert result["thresholds"]["0.9"]["tp"] == 0

    def test_mean_f1(self):
        page = _make_page([
            ((0.0, 0.0, 0.5, 0.5), [((0.1, 0.1, 0.9, 0.9), "character")]),
        ])
        result = detection_metrics([page], [page], iou_thresholds=[0.5, 0.75])
        assert result["mean_f1"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# per_class_metrics()
# ---------------------------------------------------------------------------


class TestPerClassMetrics:
    def test_per_class_perfect(self):
        page = _make_page([
            ((0.0, 0.0, 1.0, 1.0), [
                ((0.0, 0.0, 0.4, 0.4), "character"),
                ((0.5, 0.5, 0.9, 0.9), "speech_bubble"),
            ]),
        ])
        result = per_class_metrics([page], [page], iou_threshold=0.5)
        assert result["character"]["f1"] == pytest.approx(1.0)
        assert result["speech_bubble"]["f1"] == pytest.approx(1.0)

    def test_missing_class(self):
        gt = _make_page([
            ((0.0, 0.0, 1.0, 1.0), [((0.1, 0.1, 0.9, 0.9), "sfx")]),
        ])
        pred = _make_page([])
        result = per_class_metrics([pred], [gt], iou_threshold=0.5)
        assert result["sfx"]["recall"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# layout_validity_score()
# ---------------------------------------------------------------------------


class TestLayoutValidityScore:
    def test_valid_page(self, sample_page: MangaPage):
        result = layout_validity_score(sample_page)
        assert result["score"] == pytest.approx(1.0)
        assert all(result["checks"].values())

    def test_invalid_bbox_out_of_range(self):
        page = MangaPage(
            width=100,
            height=100,
            description="bad",
            panels=[Panel(bbox=(-0.5, 0.0, 1.5, 1.0), description="p")],
        )
        result = layout_validity_score(page)
        assert not result["checks"]["panel_bboxes_in_range"]
        assert result["score"] < 1.0

    def test_no_panels(self):
        page = MangaPage(width=100, height=100, description="empty")
        result = layout_validity_score(page)
        assert not result["checks"]["has_panels"]

    def test_overlapping_panels(self):
        page = MangaPage(
            width=100,
            height=100,
            description="overlap",
            panels=[
                Panel(bbox=(0.0, 0.0, 0.9, 0.9), description="a"),
                Panel(bbox=(0.05, 0.05, 0.95, 0.95), description="b"),
            ],
        )
        result = layout_validity_score(page)
        assert not result["checks"]["panels_no_excessive_overlap"]


# ---------------------------------------------------------------------------
# structural_similarity()
# ---------------------------------------------------------------------------


class TestStructuralSimilarity:
    def test_identical_pages(self, sample_page: MangaPage):
        assert structural_similarity(sample_page, sample_page) == pytest.approx(1.0)

    def test_different_pages(self):
        a = _make_page([((0.0, 0.0, 0.5, 0.5), [])])
        b = _make_page([
            ((0.0, 0.0, 0.3, 0.3), []),
            ((0.3, 0.3, 0.6, 0.6), []),
            ((0.6, 0.6, 1.0, 1.0), []),
        ])
        sim = structural_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_both_empty(self):
        a = _make_page([])
        b = _make_page([])
        assert structural_similarity(a, b) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# element_distribution()
# ---------------------------------------------------------------------------


class TestElementDistribution:
    def test_counts(self, sample_page: MangaPage):
        dist = element_distribution(sample_page)
        assert dist["total_elements"] == 3
        assert dist["distribution"]["character"]["count"] == 1
        assert dist["distribution"]["speech_bubble"]["count"] == 1
        assert dist["distribution"]["background"]["count"] == 1

    def test_empty(self):
        page = _make_page([])
        dist = element_distribution(page)
        assert dist["total_elements"] == 0


# ---------------------------------------------------------------------------
# page_summary()
# ---------------------------------------------------------------------------


class TestPageSummary:
    def test_contains_all_sections(self, sample_page: MangaPage):
        summary = page_summary(sample_page)
        assert "num_panels" in summary
        assert "num_elements" in summary
        assert "validity" in summary
        assert "element_distribution" in summary
        assert "panel_coverage" in summary
        assert "mean_panel_area" in summary
