"""Tests for mangaka.utils.bbox — bounding-box math and coordinate transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from mangaka.utils.bbox import (
    area,
    bbox_tensor_iou,
    bbox_to_retina,
    clip_bbox,
    denormalise_bbox,
    iou,
    iou_batch,
    nms,
    normalise_bbox,
    retina_to_source,
    scale_and_pad,
)


# ---------------------------------------------------------------------------
# area()
# ---------------------------------------------------------------------------


class TestArea:
    def test_standard(self):
        assert area((0.0, 0.0, 1.0, 1.0)) == pytest.approx(1.0)
        assert area((0.0, 0.0, 0.5, 0.5)) == pytest.approx(0.25)

    def test_zero_area(self):
        assert area((0.5, 0.5, 0.5, 0.5)) == pytest.approx(0.0)
        assert area((0.0, 0.0, 1.0, 0.0)) == pytest.approx(0.0)

    def test_inverted(self):
        # x2 < x1 should give 0 due to max(0, ...)
        assert area((0.5, 0.0, 0.0, 1.0)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# iou()
# ---------------------------------------------------------------------------


class TestIoU:
    def test_identical_boxes(self):
        box = (0.0, 0.0, 1.0, 1.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0.0, 0.0, 0.5, 0.5)
        b = (0.6, 0.6, 1.0, 1.0)
        assert iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 0.6, 0.6)
        b = (0.4, 0.4, 1.0, 1.0)
        # intersection = 0.2 * 0.2 = 0.04
        # union = 0.36 + 0.36 - 0.04 = 0.68
        assert iou(a, b) == pytest.approx(0.04 / 0.68, abs=1e-6)

    def test_one_inside_other(self):
        outer = (0.0, 0.0, 1.0, 1.0)
        inner = (0.25, 0.25, 0.75, 0.75)
        # intersection = 0.25, union = 1.0
        assert iou(outer, inner) == pytest.approx(0.25)

    def test_zero_area_box(self):
        a = (0.5, 0.5, 0.5, 0.5)
        b = (0.0, 0.0, 1.0, 1.0)
        assert iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# iou_batch()
# ---------------------------------------------------------------------------


class TestIoUBatch:
    def test_pairwise_shape(self):
        a = np.array([[0.0, 0.0, 1.0, 1.0], [0.5, 0.5, 1.0, 1.0]])
        b = np.array([[0.0, 0.0, 0.5, 0.5], [0.25, 0.25, 0.75, 0.75], [0.0, 0.0, 1.0, 1.0]])
        result = iou_batch(a, b)
        assert result.shape == (2, 3)

    def test_single_pair(self):
        a = np.array([[0.0, 0.0, 1.0, 1.0]])
        b = np.array([[0.0, 0.0, 1.0, 1.0]])
        result = iou_batch(a, b)
        assert result[0, 0] == pytest.approx(1.0)

    def test_no_overlap(self):
        a = np.array([[0.0, 0.0, 0.4, 0.4]])
        b = np.array([[0.6, 0.6, 1.0, 1.0]])
        result = iou_batch(a, b)
        assert result[0, 0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# nms()
# ---------------------------------------------------------------------------


class TestNMS:
    def test_suppresses_overlapping(self):
        boxes = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.05, 0.05, 0.55, 0.55],  # heavy overlap with box 0
            [0.7, 0.7, 1.0, 1.0],      # no overlap
        ])
        scores = np.array([0.9, 0.8, 0.7])
        kept = nms(boxes, scores, threshold=0.3)
        assert 0 in kept
        assert 2 in kept
        assert 1 not in kept

    def test_no_suppression(self):
        boxes = np.array([
            [0.0, 0.0, 0.3, 0.3],
            [0.5, 0.5, 0.8, 0.8],
        ])
        scores = np.array([0.9, 0.8])
        kept = nms(boxes, scores, threshold=0.5)
        assert len(kept) == 2

    def test_all_same_score(self):
        boxes = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.05, 0.05, 0.55, 0.55],
        ])
        scores = np.array([0.5, 0.5])
        kept = nms(boxes, scores, threshold=0.3)
        assert len(kept) >= 1

    def test_empty(self):
        boxes = np.array([]).reshape(0, 4)
        scores = np.array([])
        kept = nms(boxes, scores)
        assert len(kept) == 0


# ---------------------------------------------------------------------------
# normalise / denormalise roundtrip
# ---------------------------------------------------------------------------


class TestNormaliseDenormalise:
    def test_roundtrip(self):
        bbox = (100, 200, 300, 400)
        w, h = 800, 600
        normed = normalise_bbox(bbox, w, h)
        restored = denormalise_bbox(normed, w, h)
        assert restored == bbox

    def test_edge_values(self):
        normed = normalise_bbox((0, 0, 800, 600), 800, 600)
        assert normed == pytest.approx((0.0, 0.0, 1.0, 1.0))


# ---------------------------------------------------------------------------
# clip_bbox()
# ---------------------------------------------------------------------------


class TestClipBbox:
    def test_within_bounds(self):
        bbox = (0.2, 0.3, 0.7, 0.8)
        assert clip_bbox(bbox) == bbox

    def test_outside_bounds(self):
        bbox = (-0.1, -0.2, 1.3, 1.5)
        clipped = clip_bbox(bbox)
        assert clipped == (0.0, 0.0, 1.0, 1.0)

    def test_custom_range(self):
        bbox = (-5.0, -5.0, 15.0, 15.0)
        clipped = clip_bbox(bbox, min_val=0.0, max_val=10.0)
        assert clipped == (0.0, 0.0, 10.0, 10.0)


# ---------------------------------------------------------------------------
# scale_and_pad()
# ---------------------------------------------------------------------------


class TestScaleAndPad:
    def test_square_image(self):
        img = Image.new("RGB", (512, 512), (128, 128, 128))
        padded, scale, ox, oy = scale_and_pad(img, retina_size=1024)
        assert padded.size == (1024, 1024)
        assert scale == pytest.approx(2.0)
        assert ox == 0
        assert oy == 0

    def test_landscape_image(self):
        img = Image.new("RGB", (1000, 500))
        padded, scale, ox, oy = scale_and_pad(img, retina_size=1024)
        assert padded.size == (1024, 1024)
        # Width is the constraining dimension
        assert scale == pytest.approx(1024 / 1000)
        assert ox == 0
        assert oy > 0  # vertical padding

    def test_portrait_image(self):
        img = Image.new("RGB", (500, 1000))
        padded, scale, ox, oy = scale_and_pad(img, retina_size=1024)
        assert padded.size == (1024, 1024)
        assert scale == pytest.approx(1024 / 1000)
        assert ox > 0  # horizontal padding
        assert oy == 0


# ---------------------------------------------------------------------------
# bbox_to_retina / retina_to_source roundtrip
# ---------------------------------------------------------------------------


class TestRetinaMapping:
    def test_roundtrip_consistency(self):
        bbox = (100, 200, 300, 400)
        scale = 2.0
        ox, oy = 50, 100
        retina = bbox_to_retina(bbox, scale, ox, oy)
        restored = retina_to_source(retina, scale, ox, oy)
        assert restored == bbox


# ---------------------------------------------------------------------------
# bbox_tensor_iou (torch)
# ---------------------------------------------------------------------------


class TestBboxTensorIoU:
    def test_matches_numpy(self):
        a_np = np.array([[0.0, 0.0, 0.5, 0.5], [0.3, 0.3, 0.8, 0.8]])
        b_np = np.array([[0.25, 0.25, 0.75, 0.75]])
        expected = iou_batch(a_np, b_np)

        a_t = torch.tensor(a_np, dtype=torch.float32)
        b_t = torch.tensor(b_np, dtype=torch.float32)
        result = bbox_tensor_iou(a_t, b_t).numpy()
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_identical(self):
        a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        result = bbox_tensor_iou(a, a)
        assert result[0, 0].item() == pytest.approx(1.0)
