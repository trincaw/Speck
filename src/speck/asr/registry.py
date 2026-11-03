"""Engine factory and registry for model-agnostic speech recognition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from speck.asr.base import BaseASREngine
from speck.config import get_device, get_dtype_str, get_model_name
from speck.asr.engines.parakeet import ParakeetASR
from speck.asr.engines.whisper import WhisperASR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    """A built-in engine. Adding one = an engine module + one entry in ENGINE_SPECS."""
    id: str
    cls: type[BaseASREngine]
    default_model_id: str
    aliases: tuple[str, ...]
    keywords: tuple[str, ...]  # substrings routing unregistered checkpoints (e.g. new HF repos) here
    name: str
    architecture: str
    type: str
    description: str
    alias_model_ids: tuple[tuple[str, str], ...] = ()  # alias -> checkpoint, for aliases naming a model size

    def model_id_for(self, requested: str) -> str:
        """Use the requested string as model id when it's a concrete checkpoint, else the default."""
        mapped = dict(self.alias_model_ids).get(requested)
        if mapped:
            return mapped
        return requested if "/" in requested else self.default_model_id


DEFAULT_ENGINE = "parakeet"

ENGINE_SPECS: list[EngineSpec] = [
    EngineSpec(
        id="parakeet",
        cls=ParakeetASR,
        default_model_id="nvidia/parakeet-tdt-0.6b-v3",
        aliases=("parakeet", "parakeet-tdt", "nvidia/parakeet-tdt-0.6b-v3"),
        keywords=("parakeet",),
        name="NVIDIA Parakeet-TDT v3 (0.6B)",
        architecture="FastConformer-TDT",
        type="local_offline",
        description="Default. Most accurate and fastest; 25 European languages, auto-detected.",
    ),
    EngineSpec(
        id="whisper",
        cls=WhisperASR,
        default_model_id="large-v3-turbo",
        aliases=("whisper", "faster-whisper", "whisper-large-v3-turbo"),
        keywords=("whisper",),
        name="Whisper (faster-whisper)",
        architecture="Transformer encoder-decoder (CTranslate2)",
        type="local_offline",
        description="99 languages, language lock, beam search and hotword biasing.",
        alias_model_ids=(("whisper-large-v3-turbo", "large-v3-turbo"),),
    ),
]

_SPEC_BY_ID: dict[str, EngineSpec] = {spec.id: spec for spec in ENGINE_SPECS}
_SPEC_BY_ALIAS: dict[str, EngineSpec] = {alias: spec for spec in ENGINE_SPECS for alias in spec.aliases}

# alias -> engine class; extended by register_engine()
_ENGINE_REGISTRY: dict[str, type[BaseASREngine]] = {alias: spec.cls for alias, spec in _SPEC_BY_ALIAS.items()}

_DTYPES = {
    "float16": torch.float16, "fp16": torch.float16,
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float32": torch.float32, "fp32": torch.float32,
}


def register_engine(alias: str, engine_cls: type[BaseASREngine]) -> None:
    """Register a custom ASR engine class under an alias."""
    _ENGINE_REGISTRY[alias.lower()] = engine_cls
    logger.info("Registered engine '%s' -> %s", alias, engine_cls.__name__)


def resolve_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    return _DTYPES.get(dtype_str.lower().strip()) if dtype_str else None


def _resolve_spec(key: str) -> Optional[EngineSpec]:
    if key in _SPEC_BY_ALIAS:
        return _SPEC_BY_ALIAS[key]
    return next((spec for spec in ENGINE_SPECS if any(kw in key for kw in spec.keywords)), None)


def get_spec(model: Optional[str] = None) -> EngineSpec:
    """Built-in spec for an alias or checkpoint (default: SPECK_MODEL), falling back to the default engine."""
    return _resolve_spec((model or get_model_name()).strip().lower()) or _SPEC_BY_ID[DEFAULT_ENGINE]


def get_engine(
    model: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> BaseASREngine:
    """Instantiate an ASR engine by name/alias, defaulting to SPECK_MODEL from .env.

    Args:
        model: Alias or checkpoint (e.g. 'parakeet', 'whisper', 'nvidia/...').
        device: 'cuda', 'cpu', or None for auto.
        dtype: torch dtype, or None for SPECK_DTYPE / auto.
        kwargs: Engine-specific constructor arguments.
    """
    requested = (model or get_model_name()).strip()
    key = requested.lower()
    device = device or get_device()
    dtype = dtype or resolve_dtype(get_dtype_str())

    registered = _ENGINE_REGISTRY.get(key)
    spec = _resolve_spec(key)
    if registered is not None and (spec is None or registered is not spec.cls):
        # Custom engine registered via register_engine()
        return registered(model_id=requested, device=device, dtype=dtype, **kwargs)

    if spec is None:
        logger.warning("Unrecognized model '%s'. Defaulting to %s.", requested, DEFAULT_ENGINE)
        spec = _SPEC_BY_ID[DEFAULT_ENGINE]
    return spec.cls(model_id=spec.model_id_for(requested), device=device, dtype=dtype, **kwargs)


def list_available_models() -> list[dict[str, Any]]:
    """Metadata for built-in engines, flagging the one selected by SPECK_MODEL."""
    active = _resolve_spec(get_model_name().lower())
    return [
        {
            "id": spec.id,
            "name": spec.name,
            "model_id": spec.default_model_id,
            "architecture": spec.architecture,
            "type": spec.type,
            "description": spec.description,
            "is_active": spec is active,
        }
        for spec in ENGINE_SPECS
    ]
