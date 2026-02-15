"""
OpenAI GPT-4o vision annotator for manga pages.
"""

from __future__ import annotations

import os

from PIL import Image

from mangaka.annotator.base import Annotator, image_to_base64


class OpenAIAnnotator(Annotator):
    """Annotate manga pages using OpenAI's vision API (GPT-4o)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4o",
        max_tokens: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
        )
        self.model = model
        self.max_tokens = max_tokens

    async def _call_vlm(
        self,
        image: Image.Image,
        prompt: str,
        system: str | None = None,
    ) -> str:
        b64 = image_to_base64(image, fmt="PNG")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64}",
                        "detail": "high",
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
