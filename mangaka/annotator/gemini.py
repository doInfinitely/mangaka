"""
Google Gemini vision annotator for manga pages.
"""

from __future__ import annotations

import os

from PIL import Image

from mangaka.annotator.base import Annotator


class GeminiAnnotator(Annotator):
    """Annotate manga pages using Google Gemini's vision API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-2.0-flash",
        **kwargs,
    ):
        super().__init__(**kwargs)
        from google import genai

        self._genai = genai
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY"),
        )
        self.model = model

    async def _call_vlm(
        self,
        image: Image.Image,
        prompt: str,
        system: str | None = None,
    ) -> str:
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=[image, full_prompt],
        )
        return response.text
