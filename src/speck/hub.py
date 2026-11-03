"""Model assets fetched from the HuggingFace Hub, and how they are resolved to local files."""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Speaker embeddings. TitaNet keeps a stable similarity scale across utterance lengths (CAM++ swings
# from 0.2 to 0.8 between 1.5 s and 3 s clips), which a single online threshold depends on.
EMBEDDING_REPO = "csukuangfj/speaker-embedding-models"
EMBEDDING_FILE = "nemo_en_titanet_small.onnx"

# Speaker segmentation for offline diarization
PYANNOTE_SEGMENTATION_REPO = "csukuangfj/sherpa-onnx-pyannote-segmentation-3-0"
PYANNOTE_SEGMENTATION_FILE = "model.int8.onnx"

# Voice activity detection
VAD_REPO = "csukuangfj/vad"
VAD_FILE = "silero_vad.onnx"


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
