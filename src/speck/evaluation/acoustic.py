"""Acoustic TTS metrics: Mel-Cepstral Distortion, F0 prosody (RMSE, PCC, VDE, GPE) and speaker similarity (SECS)."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union

import librosa
import librosa.sequence
import numpy as np
from scipy.fftpack import dct

from speck.audio import to_mono_float32

# Scaling factor for Mel-Cepstral Distortion in dB: (10 * sqrt(2)) / ln(10)
MCD_FACTOR = (10.0 * math.sqrt(2.0)) / math.log(10.0)  # ~6.14185

AudioLike = Union[str, np.ndarray]


def load_audio(audio: AudioLike, sample_rate: int) -> np.ndarray:
    """Mono float32 samples from a path (resampled to `sample_rate`) or an array."""
    if isinstance(audio, str):
        return librosa.load(audio, sr=sample_rate, mono=True)[0]
    return to_mono_float32(audio)


# ==============================================================================
# ACOUSTIC EVALUATION: MEL-CEPSTRAL DISTORTION (MCD)
# ==============================================================================

def _extract_mcep(
    audio: AudioLike,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_mfcc: int = 16,
    hop_length: int = 160,
    n_fft: int = 512,
    exclude_c0: bool = True,
) -> np.ndarray:
    """
    Extracts Mel-Generalized Cepstral (MCEP) coefficients according to speech synthesis standards:
    - Mel filterbank energy envelope (n_mels=80)
    - Normalized per-frame energy to isolate spectral shape from overall loudness
    - Natural logarithm in accordance with the theoretical MCD factor: (10 * sqrt(2)) / ln(10)
    - Orthogonal DCT scaled by 1/sqrt(n_mels) to preserve L2 norm in cepstral space
    - Excludes c0 (energy coefficient)
    """
    wav = load_audio(audio, sample_rate)
    first = 1 if exclude_c0 else 0

    # Guard against completely silent / empty signals
    if len(wav) == 0 or np.std(wav) < 1e-7:
        return np.zeros((n_mfcc - first, max(1, len(wav) // hop_length)), dtype=np.float32)

    S = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=1.0)
    # Normalize per frame to eliminate global volume/gain variations (isolates timbre/spectral envelope)
    S = S / np.maximum(np.sum(S, axis=0, keepdims=True), 1e-8)
    # Floor at 80 dB dynamic range to prevent log divergence in silent frequency bands
    peak = float(np.max(S)) if np.max(S) > 0 else 1.0
    log_S = np.log(np.maximum(S, peak * 1e-4))
    cep = dct(log_S, type=2, axis=0, norm="ortho") / np.sqrt(S.shape[0])
    return cep[first:n_mfcc].astype(np.float32)


def compute_mcd(
    ref_audio: AudioLike,
    syn_audio: AudioLike,
    sample_rate: int = 16000,
    n_mels: int = 80,
    n_mfcc: int = 16,
    hop_length: int = 160,
    n_fft: int = 512,
    exclude_c0: bool = True,
) -> float:
    """
    Computes Mel-Cepstral Distortion (MCD) in decibels (dB) using Dynamic Time Warping (DTW):
        MCD = (10 * sqrt(2) / ln(10)) * (1 / K) * sum_{k=1}^K ||c_ref(p_k[0]) - c_syn(p_k[1])||_2

    Lower is better:
    - 0.0 dB: Identical spectral envelope
    - < 5.0 dB: High-fidelity natural speech synthesis
    - 5.0 - 7.5 dB: Intelligible TTS with noticeable prosodic/acoustic variation
    - > 8.0 dB: Noticeable spectral divergence
    """
    mcep_ref = _extract_mcep(ref_audio, sample_rate, n_mels, n_mfcc, hop_length, n_fft, exclude_c0)
    mcep_syn = _extract_mcep(syn_audio, sample_rate, n_mels, n_mfcc, hop_length, n_fft, exclude_c0)
    if mcep_ref.shape == mcep_syn.shape and np.allclose(mcep_ref, mcep_syn, atol=1e-5):
        return 0.0
    _, wp = librosa.sequence.dtw(X=mcep_ref, Y=mcep_syn, metric="euclidean")
    diff = mcep_ref[:, wp[:, 0]] - mcep_syn[:, wp[:, 1]]
    return float(MCD_FACTOR * np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))


# ==============================================================================
# PROSODY & INTONATION: F0 PITCH METRICS (RMSE, PCC, VDE, GPE)
# ==============================================================================

def _extract_f0(
    wav: np.ndarray,
    sample_rate: int = 16000,
    hop_length: int = 160,
    fmin: float = 50.0,
    fmax: float = 500.0,
) -> np.ndarray:
    """Extracts pitch contour with energy-gated voicing detection."""
    rms = librosa.feature.rms(y=wav, hop_length=hop_length, frame_length=512)[0]
    max_rms = float(np.max(rms)) if len(rms) > 0 else 0.0
    if max_rms < 1e-4:
        return np.zeros(len(rms), dtype=np.float32)
    try:
        f0 = librosa.yin(wav, fmin=fmin, fmax=fmax, sr=sample_rate, hop_length=hop_length)
    except Exception:
        return np.zeros(len(rms), dtype=np.float32)

    f0 = np.nan_to_num(f0, nan=0.0, posinf=0.0, neginf=0.0)
    n = min(len(rms), len(f0))
    rms, f0 = rms[:n], f0[:n]
    # Voicing condition: frame energy above noise floor and within valid human pitch bounds
    is_voiced = (rms > 0.02 * max_rms) & (rms > 1e-4) & (f0 > fmin + 1.0) & (f0 < fmax - 1.0)
    f0[~is_voiced] = 0.0
    return f0.astype(np.float32)


def _f0_result(frames: int = 0, voiced: int = 0, **values: float) -> Dict[str, Any]:
    return {
        "f0_rmse_hz": 0.0, "f0_rmse_cents": 0.0, "f0_pcc": 0.0, "vde": 0.0, "gpe": 0.0,
        **values, "total_aligned_frames": frames, "mutually_voiced_frames": voiced,
    }


def compute_f0_metrics(
    ref_audio: AudioLike,
    syn_audio: AudioLike,
    sample_rate: int = 16000,
    fmin: float = 50.0,
    fmax: float = 500.0,
    hop_length: int = 160,
    n_fft: int = 512,
) -> Dict[str, Any]:
    """
    Computes Fundamental Frequency (F0) prosody metrics aligned via Dynamic Time Warping:
    - F0-RMSE: Root Mean Square Error on voiced frames (in Hz)
    - F0-Cents-RMSE: Pitch error in musical cents (1200 * log2(F0_syn / F0_ref))
    - F0-PCC: Pearson Correlation Coefficient of voiced pitch trajectories (-1.0 to 1.0)
    - VDE: Voicing Decision Error (fraction of frames with voiced/unvoiced mismatch)
    - GPE: Gross Pitch Error (fraction of voiced frames with relative error > 20%)
    """
    wav_ref, wav_syn = load_audio(ref_audio, sample_rate), load_audio(syn_audio, sample_rate)
    if len(wav_ref) == len(wav_syn) and np.allclose(wav_ref, wav_syn, atol=1e-5):
        frames = len(wav_ref) // hop_length
        return _f0_result(frames, frames, f0_pcc=1.0)

    f0_ref = _extract_f0(wav_ref, sample_rate=sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    f0_syn = _extract_f0(wav_syn, sample_rate=sample_rate, hop_length=hop_length, fmin=fmin, fmax=fmax)
    # Alignment via MCEP DTW
    mcep_ref = _extract_mcep(wav_ref, sample_rate=sample_rate, hop_length=hop_length, n_fft=n_fft)
    mcep_syn = _extract_mcep(wav_syn, sample_rate=sample_rate, hop_length=hop_length, n_fft=n_fft)
    n_ref, n_syn = min(mcep_ref.shape[1], len(f0_ref)), min(mcep_syn.shape[1], len(f0_syn))
    if n_ref == 0 or n_syn == 0:
        return _f0_result()

    _, wp = librosa.sequence.dtw(X=mcep_ref[:, :n_ref], Y=mcep_syn[:, :n_syn], metric="euclidean")
    f0_ref_aligned, f0_syn_aligned = f0_ref[:n_ref][wp[:, 0]], f0_syn[:n_syn][wp[:, 1]]
    total_frames = len(wp)

    # Voicing decisions (voiced when F0 > 0)
    voiced_ref, voiced_syn = f0_ref_aligned > 0.0, f0_syn_aligned > 0.0
    both_voiced = voiced_ref & voiced_syn
    vde = float(np.sum(voiced_ref ^ voiced_syn) / total_frames) if total_frames > 0 else 0.0
    voiced_count = int(np.sum(both_voiced))
    if voiced_count == 0:
        return _f0_result(total_frames, vde=vde)

    v_ref, v_syn = f0_ref_aligned[both_voiced], f0_syn_aligned[both_voiced]
    cents_diff = 1200.0 * np.log2(np.clip(v_syn / np.maximum(v_ref, 1e-6), 1e-3, 1e3))
    rel_error = np.abs(v_syn - v_ref) / np.maximum(v_ref, 1e-6)

    # Pearson Correlation Coefficient (PCC)
    std_ref, std_syn = np.std(v_ref), np.std(v_syn)
    if std_ref < 1e-5 or std_syn < 1e-5:  # constant pitch trajectory
        f0_pcc = 1.0 if np.isclose(np.mean(v_ref), np.mean(v_syn), atol=1.0) else 0.0
    else:
        cov = np.mean((v_ref - np.mean(v_ref)) * (v_syn - np.mean(v_syn)))
        f0_pcc = max(-1.0, min(1.0, float(cov / (std_ref * std_syn))))

    return _f0_result(
        total_frames,
        voiced_count,
        f0_rmse_hz=float(np.sqrt(np.mean((v_ref - v_syn) ** 2))),
        f0_rmse_cents=float(np.sqrt(np.mean(cents_diff ** 2))),
        f0_pcc=f0_pcc,
        vde=vde,
        gpe=float(np.mean(rel_error > 0.20)),  # share of voiced frames off by > 20%
    )


# ==============================================================================
# SPEAKER SIMILARITY & TIMBRE FIDELITY (SECS)
# ==============================================================================

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    cos_sim = float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-9) * (np.linalg.norm(b) + 1e-9)))
    return max(-1.0, min(1.0, cos_sim))


def compute_speaker_similarity(
    ref_audio: AudioLike,
    syn_audio: AudioLike,
    embedder: Optional[Any] = None,
    sample_rate: int = 16000,
) -> float:
    """
    Computes Speaker Embedding Cosine Similarity (SECS) between reference voice and synthesized voice.
    Returns cosine similarity in range [-1.0, 1.0].
    """
    wav_ref, wav_syn = load_audio(ref_audio, sample_rate), load_audio(syn_audio, sample_rate)
    if len(wav_ref) == len(wav_syn) and np.allclose(wav_ref, wav_syn, atol=1e-5):
        return 1.0

    if embedder is None:
        try:
            from speck.speakers.embedding import SpeakerEmbedder
            embedder = SpeakerEmbedder(sample_rate=sample_rate)
        except Exception:
            pass
    if embedder is not None:
        return _cosine(embedder.embed(wav_ref), embedder.embed(wav_syn))

    # Acoustic spectral timbre fallback
    return _cosine(*(np.mean(_extract_mcep(w, sample_rate=sample_rate, n_mfcc=20), axis=1) for w in (wav_ref, wav_syn)))
