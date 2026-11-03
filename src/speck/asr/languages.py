"""Language metadata: locale tiers and language-tag helpers shared by the ASR engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageTier:
    name: str
    description: str
    locales: list[str]


def iso639_1(language: Optional[str]) -> Optional[str]:
    """'en-US' -> 'en'; 'auto' / empty -> None."""
    if not language or language.lower() in ("auto", "none"):
        return None
    return language.split("-")[0].lower()
