"""Language metadata: locale tiers and language-tag helpers shared by the ASR engines."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LanguageTier:
    name: str
    description: str
    locales: list[str]


SUPPORTED_LANGUAGES: dict[str, LanguageTier] = {
    "transcription_ready": LanguageTier(
        name="Transcription-Ready (19 locales)",
        description="Highest accuracy multilingual ASR, ready out of the box.",
        locales=[
            "en-US", "en-GB", "es-US", "es-ES", "fr-FR", "fr-CA",
            "it-IT", "pt-BR", "pt-PT", "nl-NL", "de-DE", "tr-TR",
            "ru-RU", "ar-AR", "hi-IN", "ja-JP", "ko-KR", "vi-VN",
            "uk-UA",
        ],
    ),
    "broad_coverage": LanguageTier(
        name="Broad-Coverage (13 locales)",
        description="Production ASR across an additional 13 locales.",
        locales=[
            "pl-PL", "sv-SE", "cs-CZ", "nb-NO", "da-DK", "bg-BG",
            "fi-FI", "hr-HR", "sk-SK", "zh-CN", "hu-HU", "ro-RO",
            "et-EE",
        ],
    ),
    "adaptation_ready": LanguageTier(
        name="Adaptation-Ready (8 locales)",
        description="Recognized by the tokenizer; fine-tune on in-domain data to unlock full transcription.",
        locales=["el-GR", "lt-LT", "lv-LV", "mt-MT", "sl-SI", "he-IL", "th-TH", "nn-NO"],
    ),
}

_LANG_TAG_REGEX = re.compile(r"<([a-z]{2}-[A-Z]{2})>")


def parse_language_tag(text: str) -> tuple[str, Optional[str]]:
    """Extract and separate language tag like <en-US> or <es-ES> from output text."""
    match = _LANG_TAG_REGEX.search(text)
    if match:
        return _LANG_TAG_REGEX.sub("", text).strip(), match.group(1)
    return text.strip(), None


def iso639_1(language: Optional[str]) -> Optional[str]:
    """'en-US' -> 'en'; 'auto' / empty -> None."""
    if not language or language.lower() in ("auto", "none"):
        return None
    return language.split("-")[0].lower()
