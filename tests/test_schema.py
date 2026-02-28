"""Tests for mangaka.schema — MangaPage / Panel / MangaElement models."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mangaka.schema import (
    ELEMENT_TYPE_TO_ID,
    ELEMENT_TYPES,
    ID_TO_ELEMENT_TYPE,
    NUM_ELEMENT_TYPES,
    MangaElement,
    MangaPage,
    Panel,
)


# ---------------------------------------------------------------------------
# Creation / valid data
# ---------------------------------------------------------------------------


class TestMangaElementCreation:
    def test_basic_creation(self, sample_element: MangaElement):
        assert sample_element.type == "character"
        assert sample_element.bbox == (0.1, 0.1, 0.5, 0.9)
        assert sample_element.description == "A young samurai in black robes"
        assert sample_element.text is None

    def test_with_text(self, sample_speech_bubble: MangaElement):
        assert sample_speech_bubble.type == "speech_bubble"
        assert sample_speech_bubble.text == "What are you doing here?!"

    @pytest.mark.parametrize("elem_type", ELEMENT_TYPES)
    def test_all_element_types(self, elem_type: str):
        elem = MangaElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            type=elem_type,
            description=f"A {elem_type} element",
        )
        assert elem.type == elem_type

    def test_optional_text_none(self):
        elem = MangaElement(
            bbox=(0.0, 0.0, 1.0, 1.0),
            type="character",
            description="hero",
        )
        assert elem.text is None


class TestPanelCreation:
    def test_basic_creation(self, sample_panel: Panel):
        assert sample_panel.bbox == (0.0, 0.0, 0.5, 0.5)
        assert len(sample_panel.elements) == 2

    def test_empty_elements(self):
        panel = Panel(
            bbox=(0.0, 0.0, 1.0, 1.0),
            description="Empty panel",
        )
        assert panel.elements == []


class TestMangaPageCreation:
    def test_basic_creation(self, sample_page: MangaPage):
        assert sample_page.width == 1654
        assert sample_page.height == 2339
        assert sample_page.style is not None

    def test_optional_style_none(self):
        page = MangaPage(
            width=800,
            height=600,
            description="Test page",
        )
        assert page.style is None
        assert page.panels == []

    def test_empty_panels(self):
        page = MangaPage(
            width=800,
            height=600,
            description="Empty page",
            panels=[],
        )
        assert page.num_panels == 0
        assert page.num_elements == 0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_num_panels(self, sample_page: MangaPage):
        assert sample_page.num_panels == 2

    def test_num_elements(self, sample_page: MangaPage):
        # Panel 1 has 2 elements (character + speech_bubble), Panel 2 has 1 (background)
        assert sample_page.num_elements == 3


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


class TestSerialisationRoundtrip:
    def test_to_json_returns_string(self, sample_page: MangaPage):
        text = sample_page.to_json()
        assert isinstance(text, str)
        data = json.loads(text)
        assert data["width"] == 1654

    def test_from_json_string(self, sample_page: MangaPage):
        text = sample_page.to_json()
        restored = MangaPage.from_json(text)
        assert restored.width == sample_page.width
        assert restored.num_panels == sample_page.num_panels
        assert restored.num_elements == sample_page.num_elements
        assert restored.description == sample_page.description

    def test_file_roundtrip(self, sample_page: MangaPage, tmp_path: Path):
        path = tmp_path / "page.json"
        sample_page.to_json(path)
        assert path.exists()
        restored = MangaPage.from_json(path)
        assert restored.width == sample_page.width
        assert restored.num_panels == sample_page.num_panels
        assert restored.panels[0].elements[0].type == "character"

    def test_preserves_text_fields(self, sample_page: MangaPage):
        text = sample_page.to_json()
        restored = MangaPage.from_json(text)
        bubble = restored.panels[0].elements[1]
        assert bubble.text == "What are you doing here?!"

    def test_preserves_none_text(self, sample_element: MangaElement):
        page = MangaPage(
            width=100,
            height=100,
            description="test",
            panels=[
                Panel(
                    bbox=(0.0, 0.0, 1.0, 1.0),
                    description="p",
                    elements=[sample_element],
                )
            ],
        )
        text = page.to_json()
        restored = MangaPage.from_json(text)
        assert restored.panels[0].elements[0].text is None


# ---------------------------------------------------------------------------
# Type mappings
# ---------------------------------------------------------------------------


class TestTypeMappings:
    def test_element_type_count(self):
        assert NUM_ELEMENT_TYPES == 6

    def test_type_to_id_coverage(self):
        assert len(ELEMENT_TYPE_TO_ID) == NUM_ELEMENT_TYPES
        for t in ELEMENT_TYPES:
            assert t in ELEMENT_TYPE_TO_ID

    def test_id_to_type_coverage(self):
        assert len(ID_TO_ELEMENT_TYPE) == NUM_ELEMENT_TYPES
        for i in range(NUM_ELEMENT_TYPES):
            assert i in ID_TO_ELEMENT_TYPE

    def test_roundtrip(self):
        for t in ELEMENT_TYPES:
            assert ID_TO_ELEMENT_TYPE[ELEMENT_TYPE_TO_ID[t]] == t
