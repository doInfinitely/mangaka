"""Tests for mangaka.infiller.layout — spatial conditioning map rendering."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from mangaka.infiller.layout import (
    NUM_LAYOUT_CHANNELS,
    layout_to_pil,
    render_element_layout,
    render_element_mask_in_panel,
    render_full_mask,
    render_page_layout,
    render_panel_layout,
    render_panel_mask,
)
from mangaka.schema import (
    ELEMENT_TYPE_TO_ID,
    MangaElement,
    MangaPage,
    Panel,
)


RESOLUTION = 256  # use a small resolution for faster tests


# ---------------------------------------------------------------------------
# render_page_layout()
# ---------------------------------------------------------------------------


class TestRenderPageLayout:
    def test_shape(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        assert layout.shape == (NUM_LAYOUT_CHANNELS, RESOLUTION, RESOLUTION)

    def test_dtype(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        assert layout.dtype == np.float32

    def test_non_zero_border_channel(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        assert layout[0].max() > 0, "border channel should have panel borders"

    def test_non_zero_fill_channel(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        assert layout[1].max() > 0, "fill channel should have panel fills"

    def test_type_channels_activated(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        # The page has character, speech_bubble, and background elements
        char_ch = 2 + ELEMENT_TYPE_TO_ID["character"]
        assert layout[char_ch].max() > 0

    def test_empty_page(self):
        page = MangaPage(width=100, height=100, description="empty")
        layout = render_page_layout(page, resolution=RESOLUTION)
        assert layout.shape == (NUM_LAYOUT_CHANNELS, RESOLUTION, RESOLUTION)
        # No panels, so borders and fills should be zero
        assert layout[0].max() == 0
        assert layout[1].max() == 0


# ---------------------------------------------------------------------------
# render_panel_layout()
# ---------------------------------------------------------------------------


class TestRenderPanelLayout:
    def test_shape(self, sample_panel: Panel):
        layout = render_panel_layout(sample_panel, resolution=RESOLUTION)
        assert layout.shape == (NUM_LAYOUT_CHANNELS, RESOLUTION, RESOLUTION)

    def test_border_channel(self, sample_panel: Panel):
        layout = render_panel_layout(sample_panel, resolution=RESOLUTION)
        assert layout[0].max() > 0

    def test_fill_channel(self, sample_panel: Panel):
        layout = render_panel_layout(sample_panel, resolution=RESOLUTION)
        assert layout[1].max() > 0

    def test_type_channels(self, sample_panel: Panel):
        layout = render_panel_layout(sample_panel, resolution=RESOLUTION)
        char_ch = 2 + ELEMENT_TYPE_TO_ID["character"]
        bubble_ch = 2 + ELEMENT_TYPE_TO_ID["speech_bubble"]
        assert layout[char_ch].max() > 0
        assert layout[bubble_ch].max() > 0


# ---------------------------------------------------------------------------
# render_element_layout()
# ---------------------------------------------------------------------------


class TestRenderElementLayout:
    def test_shape(self, sample_element: MangaElement):
        layout = render_element_layout(sample_element, resolution=RESOLUTION)
        assert layout.shape == (NUM_LAYOUT_CHANNELS, RESOLUTION, RESOLUTION)

    def test_full_fill(self, sample_element: MangaElement):
        layout = render_element_layout(sample_element, resolution=RESOLUTION)
        assert layout[1].min() == pytest.approx(1.0)  # entire channel is filled

    def test_correct_type_channel(self, sample_element: MangaElement):
        layout = render_element_layout(sample_element, resolution=RESOLUTION)
        type_idx = ELEMENT_TYPE_TO_ID[sample_element.type]
        assert layout[2 + type_idx].min() == pytest.approx(1.0)

    def test_other_type_channels_zero(self, sample_element: MangaElement):
        layout = render_element_layout(sample_element, resolution=RESOLUTION)
        type_idx = ELEMENT_TYPE_TO_ID[sample_element.type]
        for i in range(len(ELEMENT_TYPE_TO_ID)):
            if i != type_idx:
                assert layout[2 + i].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


class TestRenderPanelMask:
    def test_shape(self, sample_panel: Panel):
        mask = render_panel_mask(sample_panel, resolution=RESOLUTION)
        assert mask.shape == (1, RESOLUTION, RESOLUTION)

    def test_all_ones(self, sample_panel: Panel):
        mask = render_panel_mask(sample_panel, resolution=RESOLUTION)
        assert mask.min() == pytest.approx(1.0)


class TestRenderElementMaskInPanel:
    def test_shape(self, sample_panel: Panel):
        mask = render_element_mask_in_panel(sample_panel, 0, resolution=RESOLUTION)
        assert mask.shape == (1, RESOLUTION, RESOLUTION)

    def test_element_region_masked(self, sample_panel: Panel):
        mask = render_element_mask_in_panel(sample_panel, 0, resolution=RESOLUTION)
        assert mask.max() == pytest.approx(1.0)
        # Not the entire canvas
        assert mask.sum() < RESOLUTION * RESOLUTION


class TestRenderFullMask:
    def test_shape(self):
        mask = render_full_mask(resolution=RESOLUTION)
        assert mask.shape == (1, RESOLUTION, RESOLUTION)

    def test_all_ones(self):
        mask = render_full_mask(resolution=RESOLUTION)
        assert mask.min() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# layout_to_pil()
# ---------------------------------------------------------------------------


class TestLayoutToPil:
    def test_returns_pil_image(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        img = layout_to_pil(layout)
        assert isinstance(img, Image.Image)

    def test_correct_size(self, sample_page: MangaPage):
        layout = render_page_layout(sample_page, resolution=RESOLUTION)
        img = layout_to_pil(layout)
        assert img.size == (RESOLUTION, RESOLUTION)


# ---------------------------------------------------------------------------
# Channel isolation
# ---------------------------------------------------------------------------


class TestChannelIsolation:
    def test_type_channels_only_activate_for_correct_types(self):
        """Each element type should only activate its own channel."""
        from mangaka.schema import ELEMENT_TYPES

        for elem_type in ELEMENT_TYPES:
            elem = MangaElement(
                bbox=(0.1, 0.1, 0.9, 0.9),
                type=elem_type,
                description=f"test {elem_type}",
            )
            layout = render_element_layout(elem, resolution=RESOLUTION)
            type_idx = ELEMENT_TYPE_TO_ID[elem_type]
            # This type channel should be active
            assert layout[2 + type_idx].max() > 0
            # Other type channels should be zero
            for other_idx in range(len(ELEMENT_TYPES)):
                if other_idx != type_idx:
                    assert layout[2 + other_idx].max() == pytest.approx(0.0), (
                        f"{elem_type} activated channel for "
                        f"{ELEMENT_TYPES[other_idx]}"
                    )
