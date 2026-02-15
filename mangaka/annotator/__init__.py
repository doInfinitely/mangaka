"""Cloud LLM annotation providers for manga pages."""

from mangaka.annotator.base import Annotator
from mangaka.annotator.claude import ClaudeAnnotator
from mangaka.annotator.openai import OpenAIAnnotator
from mangaka.annotator.gemini import GeminiAnnotator

__all__ = ["Annotator", "ClaudeAnnotator", "OpenAIAnnotator", "GeminiAnnotator"]
