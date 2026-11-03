"""Model assets fetched from the HuggingFace Hub, and how they are resolved to local files."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, TypeVar

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class OnnxModel:
    """An ONNX model file in a HuggingFace repo. Adding a model = one entry in its registry below."""
    id: str
    name: str
    repo: str
    filename: str

    def path(self, custom_path: Optional[str] = None) -> str:
        return resolve_hf_file(custom_path, self.repo, self.filename)


@dataclass(frozen=True)
class EmbeddingModel(OnnxModel):
    # Calibrated per model (scripts/speaker_benchmark.py): cosine similarity needed to accept an
    # enrolled speaker, and the clustering distance for diarization. Score scales differ across models.
    match_threshold: float = 0.35
    cluster_threshold: float = 0.7


# Speaker embeddings. TitaNet keeps a stable similarity scale across utterance lengths (CAM++ swings
# from 0.2 to 0.8 between 1.5 s and 3 s clips), which a single online threshold depends on.
# Lowest EER of the 9 candidates benchmarked in docs/models_report.md.
EMBEDDING_MODELS: dict[str, EmbeddingModel] = {
    "titanet-small": EmbeddingModel(
        "titanet-small", "NeMo TitaNet-small", "csukuangfj/speaker-embedding-models", "nemo_en_titanet_small.onnx",
        match_threshold=0.35, cluster_threshold=0.7,
    ),
}

# Speaker segmentation for offline diarization
SEGMENTATION_MODELS: dict[str, OnnxModel] = {
    "pyannote-3.0": OnnxModel(
        "pyannote-3.0", "pyannote segmentation-3.0", "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0",
        "model.int8.onnx",
    ),
}

# Voice activity detection
VAD_MODELS: dict[str, OnnxModel] = {
    "silero": OnnxModel("silero", "Silero VAD", "csukuangfj/vad", "silero_vad.onnx"),
}

DEFAULT_EMBEDDING, DEFAULT_SEGMENTATION, DEFAULT_VAD = "titanet-small", "pyannote-3.0", "silero"

M = TypeVar("M", bound=OnnxModel)


def _pick(registry: dict[str, M], name: Optional[str], default: str, kind: str) -> M:
    model = registry.get((name or default).strip().lower())
    if model is None:
        raise ValueError(f"Unknown {kind} model '{name}'. Available: {', '.join(registry)}")
    return model


def embedding_model(name: Optional[str] = None) -> EmbeddingModel:
    """Speaker-embedding model by id (default: SPECK_SPEAKER_MODEL, else TitaNet-small)."""
    from speck.config import get_speaker_model

    return _pick(EMBEDDING_MODELS, name or get_speaker_model(), DEFAULT_EMBEDDING, "speaker embedding")


def segmentation_model(name: Optional[str] = None) -> OnnxModel:
    return _pick(SEGMENTATION_MODELS, name, DEFAULT_SEGMENTATION, "segmentation")


def vad_model(name: Optional[str] = None) -> OnnxModel:
    return _pick(VAD_MODELS, name, DEFAULT_VAD, "VAD")


def default_assets() -> list[OnnxModel]:
    """The ONNX models the default configuration loads (pre-fetched by `speck setup`)."""
    return [embedding_model(), segmentation_model(), vad_model()]


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
