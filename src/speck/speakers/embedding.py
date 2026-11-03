"""Speaker embeddings (sherpa-onnx): one L2-normalized voice vector per stretch of audio."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

from speck.audio import to_mono_float32
from speck.hub import EMBEDDING_FILE, EMBEDDING_REPO, resolve_hf_file


class SpeakerEmbedder:
    """L2-normalized speaker embeddings (NeMo TitaNet-small by default)."""

    def __init__(self, model: Optional[str] = None, num_threads: int = 2, provider: str = "cpu", sample_rate: int = 16000):
        import sherpa_onnx

        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=resolve_hf_file(model, EMBEDDING_REPO, EMBEDDING_FILE),
            num_threads=num_threads,
            provider=provider,
        )
        if not cfg.validate():
            raise ValueError(f"Invalid speaker embedding config: {cfg}")
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._lock = threading.Lock()  # shared by concurrent live sessions
        self.sample_rate = sample_rate
        self.dim = self._extractor.dim

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(self.sample_rate, to_mono_float32(samples))
        stream.input_finished()
        with self._lock:
            emb = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)
