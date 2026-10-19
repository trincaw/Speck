"""Engine factory and registry for model-agnostic speech recognition."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import torch

from spek.base import BaseASREngine
from spek.config import get_device, get_dtype_str, get_model_name
from spek.engines.nemotron import NemotronASR
from spek.engines.sherpa import SherpaOnnxASR
from spek.engines.voxtral import VoxtralASR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineSpec:
    id: str
    cls: type[BaseASREngine]
    default_model_id: str
    aliases: tuple[str, ...]
    name: str
    architecture: str
    type: str
    description: str
    passthrough_ids: tuple[str, ...] = ()  # aliases that are also valid model ids

    def model_id_for(self, requested: str) -> str:
        """Use the requested string as model id when it's a concrete checkpoint, else the default."""
        return requested if "/" in requested or requested in self.passthrough_ids else self.default_model_id


ENGINE_SPECS: list[EngineSpec] = [
    EngineSpec(
        id="nemotron",
        cls=NemotronASR,
        default_model_id="nvidia/nemotron-3.5-asr-streaming-0.6b",
        aliases=("nemotron", "nemotron-3.5", "nvidia/nemotron-3.5-asr-streaming-0.6b"),
        name="NVIDIA Nemotron 3.5 ASR (0.6B)",
        architecture="Cache-Aware FastConformer-RNNT",
        type="local_streaming",
        description="High-accuracy, 40-language streaming speech recognition on PyTorch / GPU / CPU.",
    ),
    EngineSpec(
        id="voxtral-mini-transcribe-2",
        cls=VoxtralASR,
        default_model_id="voxtral-mini-latest",
        aliases=("voxtral", "voxtral-mini", "voxtral-mini-transcribe-2", "voxtral-mini-latest", "voxtral-transcribe-2"),
        name="Mistral Voxtral Mini Transcribe 2",
        architecture="Voxtral Mini Transcribe 2",
        type="api_or_local",
        description="State-of-the-art multilingual ASR with speaker diarization, timestamps, and low WER.",
        passthrough_ids=("voxtral-mini-latest", "voxtral-mini-transcribe-2"),
    ),
    EngineSpec(
        id="sherpa-onnx",
        cls=SherpaOnnxASR,
        default_model_id="sherpa-onnx",
        aliases=("sherpa", "sherpa-onnx", "sherpa_onnx", "sherpa-asr", "sherpa-onnx-asr"),
        name="Sherpa-ONNX Speech Tagging & Diarization",
        architecture="Pyannote Diarization + 3D-Speaker + Whisper ONNX",
        type="local_onnx",
        description="Auto-detects speakers (offline and live mic), tags speech per speaker via ONNX.",
    ),
]

_SPEC_BY_ALIAS: dict[str, EngineSpec] = {alias: spec for spec in ENGINE_SPECS for alias in spec.aliases}

# alias -> engine class; extended by register_engine()
_ENGINE_REGISTRY: dict[str, type[BaseASREngine]] = {alias: spec.cls for alias, spec in _SPEC_BY_ALIAS.items()}

# Substring heuristics for unregistered model strings (e.g. new HF checkpoints)
_KEYWORDS = (("sherpa", ENGINE_SPECS[2]), ("voxtral", ENGINE_SPECS[1]), ("mistral", ENGINE_SPECS[1]),
             ("nemotron", ENGINE_SPECS[0]), ("fastconformer", ENGINE_SPECS[0]))

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
    return next((spec for kw, spec in _KEYWORDS if kw in key), None)


def get_engine(
    model: Optional[str] = None,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    **kwargs: Any,
) -> BaseASREngine:
    """Instantiate an ASR engine by name/alias, defaulting to SPEK_MODEL from .env.

    Args:
        model: Alias or checkpoint (e.g. 'nemotron', 'voxtral', 'sherpa-onnx', 'nvidia/...').
        device: 'cuda', 'cpu', or None for auto.
        dtype: torch dtype, or None for SPEK_DTYPE / auto.
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
        logger.warning("Unrecognized model '%s'. Defaulting to Nemotron.", requested)
        spec = ENGINE_SPECS[0]
    return spec.cls(model_id=spec.model_id_for(requested), device=device, dtype=dtype, **kwargs)


def list_available_models() -> list[dict[str, Any]]:
    """Metadata for built-in engines, flagging the one selected by SPEK_MODEL."""
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
