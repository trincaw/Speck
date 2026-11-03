"""Evaluation metrics and benchmark pipelines for speaker recognition, ASR transcripts and TTS."""

from speck.evaluation.acoustic import MCD_FACTOR, compute_f0_metrics, compute_mcd, compute_speaker_similarity
from speck.evaluation.speaker_metrics import (
    compute_auc,
    compute_eer,
    compute_identification_metrics,
    compute_latency_percentiles,
    compute_min_dcf,
    compute_roc_curve,
    compute_rtf,
)
from speck.evaluation.text import (
    compute_cer,
    compute_mer_and_wil,
    compute_speech_text_metrics,
    compute_wer,
    normalize_english_text,
    normalize_italian_text,
    normalize_text,
)
from speck.evaluation.tts import TTSMetricsPipeline

__all__ = [
    "compute_eer",
    "compute_min_dcf",
    "compute_roc_curve",
    "compute_auc",
    "compute_identification_metrics",
    "compute_latency_percentiles",
    "compute_rtf",
    "compute_mcd",
    "compute_f0_metrics",
    "compute_speaker_similarity",
    "compute_wer",
    "compute_cer",
    "compute_mer_and_wil",
    "compute_speech_text_metrics",
    "normalize_text",
    "normalize_english_text",
    "normalize_italian_text",
    "TTSMetricsPipeline",
    "MCD_FACTOR",
]
