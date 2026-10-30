"""Configuration management and .env loader for Speck."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _find_env_file() -> Optional[Path]:
    candidates = [
        os.environ.get("SPECK_ENV_FILE"),
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _parse_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    # Inline comments are only stripped from unquoted values
    return value.split(" #", 1)[0].strip() if " #" in value else value


def load_env_file(filepath: Optional[str | Path] = None, override: bool = False) -> dict[str, str]:
    """Parse KEY=VALUE pairs from a .env file into os.environ.

    Args:
        filepath: Path to a .env file; searches SPECK_ENV_FILE and CWD if None.
        override: Overwrite variables already present in os.environ.
    """
    path = Path(filepath) if filepath else _find_env_file()
    if path is None or not path.is_file():
        return {}

    loaded: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            line = line.removeprefix("export ").strip()
            key, _, value = line.partition("=")
            key = key.strip()
            if not key:
                continue
            loaded[key] = _parse_value(value)
            if override or key not in os.environ:
                os.environ[key] = loaded[key]
    except OSError as exc:
        logger.warning("Could not read .env file at %s: %s", path, exc)
    return loaded



def env(*keys: str, default: T = None, cast: Callable[[str], T] = str) -> T:
    """Return the first non-empty environment variable among `keys`, cast to the desired type."""
    for key in keys:
        value = os.environ.get(key)
        if value:
            try:
                return cast(value.strip())
            except ValueError:
                logger.warning("Invalid value for %s=%r, using default %r", key, value, default)
                return default
    return default


def get_model_name() -> str:
    return env("SPECK_MODEL", "ASR_MODEL", default="parakeet")


def get_device() -> Optional[str]:
    return env("SPECK_DEVICE", "DEVICE")


def get_dtype_str() -> Optional[str]:
    return env("SPECK_DTYPE")


def get_mistral_api_key() -> Optional[str]:
    return env("MISTRAL_API_KEY")


def get_voxtral_model() -> str:
    return env("VOXTRAL_MODEL", "VOXTRAL_MODEL_ID", default="voxtral-mini-latest")


def get_voxtral_backend() -> str:
    backend = env("VOXTRAL_BACKEND", default="api").lower()
    return backend if backend in ("api", "local") else "api"


def get_voxtral_api_url() -> str:
    return env("VOXTRAL_API_URL", default="https://api.mistral.ai/v1/audio/transcriptions")


def get_speaker_threshold() -> float:
    """Cosine similarity needed to match a known speaker during live classification."""
    return env("SPECK_SPEAKER_THRESHOLD", default=0.35, cast=float)


def get_language() -> str:
    """Default locale ('en', 'it-IT', ...) or 'auto'; locks multilingual models to one language."""
    return env("SPECK_LANGUAGE", default="auto")


def get_hotwords() -> list[str]:
    """Domain vocabulary (names, jargon) the ASR output is biased and corrected towards."""
    raw = env("SPECK_HOTWORDS", default="")
    return [w.strip() for w in raw.split(",") if w.strip()]


def get_normalize_audio() -> bool:
    """Peak-normalize offline audio before inference (SPECK_NORMALIZE_AUDIO=0 disables it)."""
    return env("SPECK_NORMALIZE_AUDIO", default="1").lower() not in ("0", "false", "no", "off")


def get_whisper_compute_type() -> str:
    return env("SPECK_WHISPER_COMPUTE_TYPE", default="float16")


def get_beam_size() -> int:
    return env("SPECK_BEAM_SIZE", default=5, cast=int)
