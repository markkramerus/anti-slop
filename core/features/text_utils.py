"""
Text preprocessing utilities for the embedding pipeline.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Light normalization before embedding:
    - Normalize Unicode
    - Collapse excess whitespace
    - Strip leading/trailing whitespace
    """
    if not text:
        return ""
    # Normalize unicode (NFC form)
    text = unicodedata.normalize("NFC", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def truncate_text(text: str, max_chars: int = 8000) -> str:
    """Truncate text to avoid exceeding model token limits."""
    if len(text) <= max_chars:
        return text
    # Truncate at word boundary
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.9:
        truncated = truncated[:last_space]
    return truncated + " [truncated]"


def batch_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """Split texts into batches of the given size."""
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
