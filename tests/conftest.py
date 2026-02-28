"""Shared fixtures for the mangaka test suite."""

from __future__ import annotations

import pytest

from mangaka.schema import MangaElement, MangaPage, Panel


@pytest.fixture
def sample_element() -> MangaElement:
    """A character element occupying the left half of a panel."""
    return MangaElement(
        bbox=(0.1, 0.1, 0.5, 0.9),
        type="character",
        description="A young samurai in black robes",
    )


@pytest.fixture
def sample_speech_bubble() -> MangaElement:
    """A speech bubble with text."""
    return MangaElement(
        bbox=(0.5, 0.05, 0.9, 0.3),
        type="speech_bubble",
        description="Exclamation bubble with jagged border",
        text="What are you doing here?!",
    )


@pytest.fixture
def sample_panel(sample_element: MangaElement, sample_speech_bubble: MangaElement) -> Panel:
    """A panel containing a character and a speech bubble."""
    return Panel(
        bbox=(0.0, 0.0, 0.5, 0.5),
        description="Close-up of samurai reacting with surprise",
        elements=[sample_element, sample_speech_bubble],
    )


@pytest.fixture
def sample_page(sample_panel: Panel) -> MangaPage:
    """A full page with one panel."""
    second_panel = Panel(
        bbox=(0.5, 0.0, 1.0, 0.5),
        description="Wide shot of a moonlit courtyard",
        elements=[
            MangaElement(
                bbox=(0.2, 0.3, 0.8, 0.9),
                type="background",
                description="Traditional Japanese garden with stone lantern",
            ),
        ],
    )
    return MangaPage(
        width=1654,
        height=2339,
        description="A tense confrontation scene under moonlight",
        style="manga, black and white, ink drawing",
        panels=[sample_panel, second_panel],
    )
