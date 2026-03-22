"""
Llama vision annotator for manga pages.

Uses any OpenAI-compatible API hosting Llama models (Together AI, Fireworks,
etc.). Set LLAMA_API_KEY and LLAMA_BASE_URL environment variables.

Defaults to Together AI (api.together.xyz).
"""

from __future__ import annotations

import os

from PIL import Image

from mangaka.annotator.base import Annotator, image_to_base64

DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_MODEL = "meta-llama/Llama-Vision-Free"


class LlamaAnnotator(Annotator):
    """Annotate manga pages using Llama vision models via OpenAI-compatible API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 8192,
        **kwargs,
    ):
        super().__init__(**kwargs)
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=api_key or os.environ.get("LLAMA_API_KEY"),
            base_url=base_url or os.environ.get("LLAMA_BASE_URL", DEFAULT_BASE_URL),
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
