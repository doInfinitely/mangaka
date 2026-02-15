"""
Claude (Anthropic) vision annotator for manga pages.
"""

from __future__ import annotations

import os

from PIL import Image

from mangaka.annotator.base import Annotator, image_to_base64


class ClaudeAnnotator(Annotator):
    """Annotate manga pages using Claude's vision API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        import anthropic

        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
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
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)
        return response.content[0].text
