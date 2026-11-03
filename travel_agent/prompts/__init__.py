"""Prompt loader: prompts live as markdown files next to this module and are read once."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return (_DIR / f"{name}.md").read_text(encoding="utf-8").strip()
