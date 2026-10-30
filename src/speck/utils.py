"""Shared helpers and metadata: language tiers, latency presets, model file resolution."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


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

LOOKAHEAD_CONFIGS = {
    0: {"name": "ultra-low-latency", "latency_ms": 80.0, "description": "80ms chunk (0 lookahead tokens)"},
    3: {"name": "balanced", "latency_ms": 320.0, "description": "320ms chunk (3 lookahead tokens)"},
    6: {"name": "high-accuracy", "latency_ms": 560.0, "description": "560ms chunk (6 lookahead tokens - default)"},
    13: {"name": "maximum-accuracy", "latency_ms": 1120.0, "description": "1120ms chunk (13 lookahead tokens)"},
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


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Squeeze and downmix to a 1-D float32 array (channels on either axis)."""
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        # Channels are the shorter axis
        wav = wav.mean(axis=0 if wav.shape[0] < wav.shape[1] else 1)
    return wav


def normalize_audio(wav: np.ndarray, peak: float = 0.9, max_gain_db: float = 20.0) -> np.ndarray:
    """Remove DC offset and scale to a target peak; near-silent input is left alone (no noise boosting)."""
    if len(wav) == 0:
        return wav
    wav = wav - wav.mean(dtype=np.float64).astype(np.float32)
    top = float(np.abs(wav).max())
    if top < 1e-4:
        return wav
    return (wav * min(peak / top, 10 ** (max_gain_db / 20))).astype(np.float32)


def resolve_hf_file(custom_path: Optional[str], repo_id: str, filename: str) -> str:
    """Return a local model file, preferring an explicit path, then the HF cache, then a download."""
    if custom_path and os.path.isfile(custom_path):
        return custom_path

    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
    except Exception:
        logger.info("Downloading %s/%s...", repo_id, filename)
        return hf_hub_download(repo_id=repo_id, filename=filename)
