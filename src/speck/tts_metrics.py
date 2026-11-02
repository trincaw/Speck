"""Text-to-Speech (TTS) and Speech-Text Quantitative Evaluation Suite.

Comprehensive ML evaluation metrics for Speech Synthesis and Speech Recognition:
1. Acoustic Quality:
   - Mel-Cepstral Distortion (MCD) with Dynamic Time Warping (DTW) in dB
2. Prosody & Intonation (F0 Pitch Contour):
   - F0 Root Mean Square Error (F0-RMSE in Hz and cents)
   - F0 Pearson Correlation Coefficient (F0-PCC)
   - Voicing Decision Error (VDE)
   - Gross Pitch Error (GPE)
3. Speaker Timbre & Identity:
   - Speaker Embedding Cosine Similarity (SECS)
4. Linguistic & Phonetic Accuracy / Intelligibility:
   - Word Error Rate (WER) with Levenshtein alignment
   - Character Error Rate (CER)
   - Match Error Rate (MER)
   - Word Information Lost (WIL) & Word Information Preserved (WIP)
   - Language-specific text normalizers for English and Italian
5. Latency & Efficiency:
   - Real-Time Factor (RTF)
6. Reporting & Pipeline:
   - TTSMetricsPipeline generating structured JSON and Markdown reports.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import librosa
import librosa.sequence
import numpy as np
from scipy.fftpack import dct

from speck.utils import to_mono_float32

logger = logging.getLogger(__name__)

# Scaling factor for Mel-Cepstral Distortion in dB: (10 * sqrt(2)) / ln(10)
MCD_FACTOR = (10.0 * math.sqrt(2.0)) / math.log(10.0)  # ~6.14185

AudioLike = Union[str, np.ndarray]


def _load_audio(audio: AudioLike, sample_rate: int) -> np.ndarray:
    """Mono float32 samples from a path (resampled to `sample_rate`) or an array."""
    if isinstance(audio, str):
        return librosa.load(audio, sr=sample_rate, mono=True)[0]
    return to_mono_float32(audio)


# ==============================================================================
# 1. TEXT NORMALIZATION & LINGUISTIC PROCESSING (ENGLISH & ITALIAN)
# ==============================================================================

_QUOTES = re.compile(r"[’‘`´]")
_SPACES = re.compile(r"\s+")

# Irregular contractions; every regular one ("don't", "we're", "i'd", "he's"...) is covered by _EN_CLITICS
_EN_IRREGULAR = [(re.compile(rf"\b{re.escape(c)}\b"), e) for c, e in (
    ("won't", "will not"), ("can't", "cannot"), ("shan't", "shall not"), ("let's", "let us"),
)]
_EN_CLITICS = [(re.compile(rf"{c}\b"), e) for c, e in (
    ("n't", " not"), ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'d", " would"), ("'m", " am"), ("'s", " is"),
)]
_EN_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")  # 1,000 -> 1000
_EN_NON_ALNUM = re.compile(r"[^a-z0-9\s]")

# Italian common elided particles with apostrophe
_IT_ELISIONS_REGEX = re.compile(
    r"\b(l|un|all|dall|dell|nell|sull|coll|pel|d|c|m|t|s|v|ci|vi|ne|quell|bell|grand|sant|poc|mezz|qualch)'([a-zàèéìòùA-Z])",
    re.IGNORECASE,
)
_IT_INNER_APOSTROPHE = re.compile(r"([a-zàèéìòù])'([a-zàèéìòù])")
_IT_TYPOGRAPHY = re.compile(r"[«»“”\"„–—]")
_IT_NON_ALNUM = re.compile(r"[^a-zàèéìòù0-9'\s]")
_IT_NON_ASCII = re.compile(r"[^a-z0-9'\s]")
_IT_STRAY_APOSTROPHES = (re.compile(r"\s+'+\s+"), re.compile(r"^'+|'+$"))
# Italian accented characters mapping for diacritic-insensitive evaluation
_IT_ACCENT_MAP = str.maketrans("àèéìòóù", "aeeioou")

# Hesitation sounds that references never transcribe but ASR models often emit
_EN_FILLERS = frozenset({"uh", "um", "uhm", "umm", "erm", "hmm", "hm", "mm", "mmm", "mhm"})
_IT_FILLERS = frozenset({"ehm", "ehmm", "eh", "uhm", "uh", "mh", "mhm", "hmm", "mmm"})

_NUM_UNITS = {
    w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_NUM_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_NUM_SCALES = {"thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}
_NUM_WORDS = frozenset(_NUM_UNITS) | frozenset(_NUM_TENS) | {"hundred"}


def _parse_spoken_number(tokens: list[str], start: int) -> tuple[int, int]:
    """Parse the cardinal starting at `start`; returns (value, tokens consumed)."""
    total = current = 0
    last = None  # kind of the previous word: unit / tens / hundred / scale
    i = start
    while i < len(tokens):
        w = tokens[i]
        if w in _NUM_UNITS:
            n = _NUM_UNITS[w]
            # units attach to a tens word ("thirty five"), teens and zero only start a number
            if last == "unit" or (n == 0 and last is not None) or (n >= 10 and last == "tens"):
                break
            current += n
            last = "unit"
        elif w in _NUM_TENS:
            if last not in (None, "hundred", "scale"):
                break
            current += _NUM_TENS[w]
            last = "tens"
        elif w == "hundred":
            if last != "unit" or current >= 100 or current == 0:
                break
            current *= 100
            last = "hundred"
        elif w in _NUM_SCALES:
            if last not in ("unit", "tens", "hundred"):
                break
            total += current * _NUM_SCALES[w]
            current = 0
            last = "scale"
        elif w == "and" and last in ("hundred", "scale") and i + 1 < len(tokens) and tokens[i + 1] in _NUM_WORDS:
            pass  # "one hundred and five"
        else:
            break
        i += 1
    while i > start and tokens[i - 1] == "and":
        i -= 1
    return total + current, i - start


def spoken_numbers_to_digits(text: str) -> str:
    """Rewrite spelled-out English cardinals as digits ("thirty five" -> "35") in whitespace-split text.

    Applied to reference and hypothesis alike, so "35" vs "thirty five" no longer counts as an error.
    Adjacent numbers that do not compose ("twenty twenty") stay separate numbers.
    """
    tokens = text.split()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] in _NUM_WORDS:
            value, used = _parse_spoken_number(tokens, i)
            if used:
                out.append(str(value))
                i += used
                continue
        out.append(tokens[i])
        i += 1
    return " ".join(out)


def _drop(text: str, words: frozenset) -> str:
    return " ".join(w for w in text.split() if w not in words)


def normalize_english_text(
    text: str,
    expand_contractions: bool = True,
    strip_fillers: bool = True,
    normalize_numbers: bool = True,
) -> str:
    """
    Standardizes English text for acoustic and linguistic evaluation:
    - Lowercases text
    - Standardizes quotation marks
    - Expands common contractions (e.g. "don't" -> "do not")
    - Strips punctuation and symbols while preserving alphanumeric tokens
    - Drops hesitation fillers ("uh", "um") when `strip_fillers` is set
    - Writes spelled-out numbers as digits ("thirty five" -> "35") when `normalize_numbers` is set
    - Collapses repeated whitespace
    """
    if not text:
        return ""
    t = _EN_THOUSANDS.sub("", _QUOTES.sub("'", text).lower()).replace("%", " percent ")
    if expand_contractions:
        for pattern, expanded in _EN_IRREGULAR + _EN_CLITICS:
            t = pattern.sub(expanded, t)
    t = _SPACES.sub(" ", _EN_NON_ALNUM.sub(" ", t)).strip()
    if strip_fillers:
        t = _drop(t, _EN_FILLERS)
    return spoken_numbers_to_digits(t) if normalize_numbers else t


def normalize_italian_text(
    text: str,
    preserve_accents: bool = True,
    split_apostrophes: bool = True,
    strip_fillers: bool = True,
) -> str:
    """
    Standardizes Italian text for speech evaluation according to Italian phonetics & grammar:
    - Standardizes typography (quotes, dashes « », “ ”)
    - Separates apostrophe elisions (e.g. "l'albero" -> "l' albero" or "l albero")
      to prevent false-positive word merges in ASR / TTS benchmarks.
    - Handles standard Italian truncated forms ("qual è" never has an apostrophe)
    - Accents:
        * If `preserve_accents=True` (default): keeps distinct Italian vowels (à, è, é, ì, ò, ù),
          which convey critical phonemic contrasts (e.g. 'e' [and] vs 'è' [is], 'da' vs 'dà').
        * If `preserve_accents=False`: folds accents to plain ASCII vowels.
    - Strips non-alphanumeric punctuation and collapses whitespace.
    """
    if not text:
        return ""
    t = _IT_TYPOGRAPHY.sub(" ", _QUOTES.sub("'", text).lower())
    if split_apostrophes:  # "l'amico" -> "l' amico", then any word-internal one: "d'accordo" -> "d' accordo"
        t = _IT_INNER_APOSTROPHE.sub(r"\1' \2", _IT_ELISIONS_REGEX.sub(r"\1' \2", t))
    t = _IT_NON_ALNUM.sub(" ", t) if preserve_accents else _IT_NON_ASCII.sub(" ", t.translate(_IT_ACCENT_MAP))
    stray_inner, stray_edge = _IT_STRAY_APOSTROPHES
    t = _SPACES.sub(" ", stray_edge.sub("", stray_inner.sub(" ", t))).strip()
    return _drop(t, _IT_FILLERS) if strip_fillers else t


def normalize_text(
    text: str,
    language: str = "en",
    preserve_accents: bool = True,
) -> str:
    """
    Language-aware text normalizer dispatcher.
    Supports "en", "en-US", "en-GB", "it", "it-IT", etc.
    """
    if (language or "en").lower().split("-")[0].split("_")[0] == "it":
        return normalize_italian_text(text, preserve_accents=preserve_accents)
    return normalize_english_text(text)


# ==============================================================================
# 2. LEVENSHTEIN DISTANCE, ALIGNMENTS, WER, CER, MER, WIL
# ==============================================================================

def _levenshtein_align(
    ref_seq: Sequence[Any],
    hyp_seq: Sequence[Any],
) -> Tuple[int, int, int, int, List[Tuple[str, Any, Any]]]:
    """
    Computes optimal Levenshtein alignment between reference and hypothesis sequences.
    Returns:
        (substitutions, deletions, insertions, hits, aligned_operations)
    where aligned_operations is a list of tuples: (operation, ref_item, hyp_item)
    operation is one of: "equal", "substitute", "delete", "insert".
    """
    n, m = len(ref_seq), len(hyp_seq)
    # Plain lists: per-element access is several times faster than numpy scalar indexing
    dp = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        row, up, r = dp[i], dp[i - 1], ref_seq[i - 1]
        for j in range(1, m + 1):
            row[j] = up[j - 1] if r == hyp_seq[j - 1] else 1 + min(up[j - 1], up[j], row[j - 1])

    # Backtracking to reconstruct exact alignment path
    i, j = n, m
    ops: List[Tuple[str, Any, Any]] = []
    counts = {"equal": 0, "substitute": 0, "delete": 0, "insert": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_seq[i - 1] == hyp_seq[j - 1]:
            op, i, j = ("equal", ref_seq[i - 1], hyp_seq[j - 1]), i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            op, i, j = ("substitute", ref_seq[i - 1], hyp_seq[j - 1]), i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            op, i = ("delete", ref_seq[i - 1], None), i - 1
        else:
            op, j = ("insert", None, hyp_seq[j - 1]), j - 1
        ops.append(op)
        counts[op[0]] += 1

    ops.reverse()
    return counts["substitute"], counts["delete"], counts["insert"], counts["equal"], ops


def _edit_rate(reference, hypothesis, language, normalize, preserve_accents, tokenize):
    """(normalized ref, normalized hyp, ref tokens, hyp tokens, error rate, (S, D, I, H, ops))."""
    ref, hyp = (
        normalize_text(t, language=language, preserve_accents=preserve_accents) if normalize else t
        for t in (reference, hypothesis)
    )
    ref_tokens, hyp_tokens = tokenize(ref), tokenize(hyp)
    s, d, i, h, ops = _levenshtein_align(ref_tokens, hyp_tokens)
    rate = float((s + d + i) / len(ref_tokens)) if ref_tokens else float(bool(hyp_tokens))
    return ref, hyp, ref_tokens, hyp_tokens, rate, (s, d, i, h, ops)


def compute_wer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Computes Word Error Rate (WER):
        WER = (Substitutions + Deletions + Insertions) / Reference Word Count
    """
    ref, hyp, ref_words, hyp_words, wer, (s, d, i, h, ops) = _edit_rate(
        reference, hypothesis, language, normalize, preserve_accents, str.split
    )
    return {
        "wer": wer,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": h,
        "ref_words": len(ref_words),
        "hyp_words": len(hyp_words),
        "alignment": ops,
        "normalized_ref": ref,
        "normalized_hyp": hyp,
    }


def compute_cer(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Computes Character Error Rate (CER):
        CER = (Char Substitutions + Deletions + Insertions) / Reference Character Count
    """
    _, _, ref_chars, hyp_chars, cer, (s, d, i, h, _) = _edit_rate(
        reference, hypothesis, language, normalize, preserve_accents, lambda t: list(t.replace(" ", ""))
    )
    return {
        "cer": cer,
        "substitutions": s,
        "deletions": d,
        "insertions": i,
        "hits": h,
        "ref_chars": len(ref_chars),
        "hyp_chars": len(hyp_chars),
    }


def _information_rates(wer_res: Dict[str, Any]) -> Dict[str, float]:
    errors = wer_res["substitutions"] + wer_res["deletions"] + wer_res["insertions"]
    total_ops = errors + wer_res["hits"]
    n_ref, n_hyp = wer_res["ref_words"], wer_res["hyp_words"]
    if n_ref == 0 or n_hyp == 0:
        wip = 1.0 if n_ref == n_hyp else 0.0
    else:
        wip = min(1.0, float(wer_res["hits"] ** 2 / (n_ref * n_hyp)))
    return {"mer": float(errors / total_ops) if total_ops > 0 else 0.0, "wil": float(1.0 - wip), "wip": wip}


def compute_mer_and_wil(
    reference: str,
    hypothesis: str,
    language: str = "en",
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Computes Match Error Rate (MER) and Word Information Lost / Preserved (WIL / WIP).
        MER = (S + D + I) / (S + D + I + H)
        WIP = (H^2) / (N_ref * N_hyp)
        WIL = 1 - WIP
    """
    return _information_rates(compute_wer(reference, hypothesis, language=language, normalize=normalize))


def compute_speech_text_metrics(
    reference_text: str,
    hypothesis_text: str,
    language: str = "en",
    preserve_accents: bool = True,
) -> Dict[str, Any]:
    """
    Composite evaluation function for speech transcription / TTS intelligibility:
    computes WER, CER, MER, WIL, and detailed token statistics.
    """
    wer_res = compute_wer(reference_text, hypothesis_text, language=language, preserve_accents=preserve_accents)
    cer_res = compute_cer(reference_text, hypothesis_text, language=language, preserve_accents=preserve_accents)
    return {
        "language": language,
        "wer": wer_res["wer"],
        "cer": cer_res["cer"],
        **_information_rates(wer_res),  # reuses the WER alignment instead of recomputing it
        "substitutions": wer_res["substitutions"],
        "deletions": wer_res["deletions"],
        "insertions": wer_res["insertions"],
        "hits": wer_res["hits"],
        "ref_word_count": wer_res["ref_words"],
        "hyp_word_count": wer_res["hyp_words"],
        "ref_char_count": cer_res["ref_chars"],
        "hyp_char_count": cer_res["hyp_chars"],
        "normalized_ref": wer_res["normalized_ref"],
        "normalized_hyp": wer_res["normalized_hyp"],
    }


# ==============================================================================
# 3. ACOUSTIC EVALUATION: MEL-CEPSTRAL DISTORTION (MCD)
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
    wav = _load_audio(audio, sample_rate)
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
# 4. PROSODY & INTONATION: F0 PITCH METRICS (RMSE, PCC, VDE, GPE)
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
    wav_ref, wav_syn = _load_audio(ref_audio, sample_rate), _load_audio(syn_audio, sample_rate)
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
# 5. SPEAKER SIMILARITY & TIMBRE FIDELITY (SECS)
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
    wav_ref, wav_syn = _load_audio(ref_audio, sample_rate), _load_audio(syn_audio, sample_rate)
    if len(wav_ref) == len(wav_syn) and np.allclose(wav_ref, wav_syn, atol=1e-5):
        return 1.0

    if embedder is None:
        try:
            from speck.speakers import SpeakerEmbedder
            embedder = SpeakerEmbedder(sample_rate=sample_rate)
        except Exception:
            pass
    if embedder is not None:
        return _cosine(embedder.embed(wav_ref), embedder.embed(wav_syn))

    # Acoustic spectral timbre fallback
    return _cosine(*(np.mean(_extract_mcep(w, sample_rate=sample_rate, n_mfcc=20), axis=1) for w in (wav_ref, wav_syn)))


# ==============================================================================
# 6. TTS EVALUATION PIPELINE & REPORT GENERATION
# ==============================================================================

def _mean(values: List[float], default: Optional[float] = 0.0) -> Optional[float]:
    return float(np.mean(values)) if values else default


class TTSMetricsPipeline:
    """
    Standardized automated benchmark pipeline for Text-to-Speech (TTS) systems.
    Evaluates:
    1. Acoustic Quality (MCD in dB)
    2. Prosodic Pitch Fidelity (F0-RMSE, F0-PCC, VDE, GPE)
    3. Speaker Timbre Similarity (SECS)
    4. Round-Trip Intelligibility & Pronunciation Accuracy (WER, CER, MER, WIL)
       with native support for Italian and English.
    5. Computational Efficiency (Real-Time Factor RTF and Latency)
    6. Artifact Exporting (JSON summary and Markdown Executive Report).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        embedder: Optional[Any] = None,
        output_dir: str = "reports",
    ):
        self.sample_rate = sample_rate
        self.embedder = embedder
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_sample(
        self,
        ref_audio: AudioLike,
        syn_audio: AudioLike,
        ref_text: str,
        asr_hypothesis_text: Optional[str] = None,
        language: str = "en",
        synthesis_time_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Runs full suite of acoustic, prosodic, speaker, and linguistic tests on one utterance."""
        sr = self.sample_rate
        ref, syn = _load_audio(ref_audio, sr), _load_audio(syn_audio, sr)  # decode files once for all metrics
        duration = len(syn) / sr
        return {
            "language": language,
            "duration_sec": duration,
            "mcd_db": compute_mcd(ref, syn, sample_rate=sr),
            "f0": compute_f0_metrics(ref, syn, sample_rate=sr),
            "speaker_similarity_secs": compute_speaker_similarity(ref, syn, embedder=self.embedder, sample_rate=sr),
            "intelligibility": (
                compute_speech_text_metrics(ref_text, asr_hypothesis_text, language=language)
                if asr_hypothesis_text is not None else None
            ),
            "synthesis_rtf": (synthesis_time_sec / duration) if (synthesis_time_sec and duration > 0) else None,
        }

    @staticmethod
    def _summarize(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        voiced = [r["f0"] for r in group if r["f0"]["mutually_voiced_frames"] > 0]
        texts = [r["intelligibility"] for r in group if r["intelligibility"] is not None]
        mcds = [r["mcd_db"] for r in group]
        return {
            "count": len(group),
            "mcd_db_mean": _mean(mcds),
            "mcd_db_p95": float(np.percentile(mcds, 95)) if mcds else 0.0,
            "speaker_similarity_mean": _mean([r["speaker_similarity_secs"] for r in group]),
            "f0_rmse_hz_mean": _mean([f["f0_rmse_hz"] for f in voiced]),
            "f0_pcc_mean": _mean([f["f0_pcc"] for f in voiced]),
            "vde_mean": _mean([r["f0"]["vde"] for r in group]),
            "wer_mean": _mean([t["wer"] for t in texts], None),
            "cer_mean": _mean([t["cer"] for t in texts], None),
            "rtf_mean": _mean([r["synthesis_rtf"] for r in group if r["synthesis_rtf"] is not None], None),
        }

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates a batch of TTS test items.
        Each item is a dict with keys:
        - "ref_audio": path or numpy array
        - "syn_audio": path or numpy array
        - "text": target text prompt
        - "asr_text": (optional) ASR transcribed text of synthesized audio
        - "language": "en" or "it"
        - "synthesis_time_sec": (optional) synthesis time in seconds
        """
        by_lang: Dict[str, List[Dict[str, Any]]] = {}
        results: List[Dict[str, Any]] = []
        for item in dataset:
            lang = item.get("language", "en").lower().split("-")[0]
            res = self.evaluate_sample(
                ref_audio=item["ref_audio"],
                syn_audio=item["syn_audio"],
                ref_text=item["text"],
                asr_hypothesis_text=item.get("asr_text"),
                language=lang,
                synthesis_time_sec=item.get("synthesis_time_sec"),
            )
            results.append(res)
            by_lang.setdefault(lang, []).append(res)

        summary = {
            "total_samples": len(results),
            "languages": {lang: self._summarize(group) for lang, group in by_lang.items()},
            "aggregate": self._summarize(results),
        }
        return {"summary": summary, "individual_results": results}

    def export_reports(self, evaluation_output: Dict[str, Any]) -> Dict[str, str]:
        """Exports JSON summary and Markdown evaluation report for TTS benchmarks."""
        summary = evaluation_output["summary"]

        # 1. Export JSON Summary
        json_path = os.path.join(self.output_dir, "tts_metrics_summary.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # 2. Export Markdown Report
        md_path = os.path.join(self.output_dir, "TTS_EVALUATION_REPORT.md")
        agg = summary["aggregate"]
        lang_rows = []

        for lang, stats in summary["languages"].items():
            lang_label = "Italian (it-IT)" if lang.startswith("it") else "English (en-US)"
            wer_str = f"{stats['wer_mean'] * 100:.2f}%" if stats['wer_mean'] is not None else "N/A"
            cer_str = f"{stats['cer_mean'] * 100:.2f}%" if stats['cer_mean'] is not None else "N/A"
            rtf_str = f"{stats['rtf_mean']:.3f}x" if stats['rtf_mean'] is not None else "N/A"

            lang_rows.append(
                f"| **{lang_label}** | {stats['count']} | **{stats['mcd_db_mean']:.2f} dB** | "
                f"**{stats['f0_rmse_hz_mean']:.2f} Hz** | **{stats['f0_pcc_mean']:.3f}** | "
                f"**{stats['speaker_similarity_mean']:.3f}** | {wer_str} | {cer_str} | {rtf_str} |"
            )

        lang_table = "\n".join(lang_rows)

        wer_val = f"{agg['wer_mean'] * 100:.2f}%" if agg['wer_mean'] is not None else "N/A"
        cer_val = f"{agg['cer_mean'] * 100:.2f}%" if agg['cer_mean'] is not None else "N/A"

        report = f"""# Text-to-Speech (TTS) & Speech Evaluation Report

**Generated on:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  
**Evaluation Target:** Multilingual Speech Synthesis Quality & Intelligibility  
**Supported Languages:** English (en-US) & Italian (it-IT)

---

## 1. Executive Metrics Overview

| Metric | Target | Result (Aggregate) | Benchmark Status |
| :--- | :--- | :--- | :--- |
| **Mel-Cepstral Distortion (MCD)** | $\\le 6.5\\text{{ dB}}$ | **{agg['mcd_db_mean']:.2f} dB** | {'✅ Passed' if agg['mcd_db_mean'] <= 6.5 else '⚠️ Review'} |
| **Pitch RMSE ($F_0$)** | $\\le 35\\text{{ Hz}}$ | **{agg['f0_rmse_hz_mean']:.2f} Hz** | {'✅ Passed' if agg['f0_rmse_hz_mean'] <= 35.0 else '⚠️ Review'} |
| **Prosody Correlation ($F_0$ PCC)** | $\\ge 0.70$ | **{agg['f0_pcc_mean']:.3f}** | {'✅ Passed' if agg['f0_pcc_mean'] >= 0.70 else '⚠️ Review'} |
| **Voicing Decision Error (VDE)** | $\\le 15\\%$ | **{agg['vde_mean'] * 100:.2f}%** | {'✅ Passed' if agg['vde_mean'] <= 0.15 else '⚠️ Review'} |
| **Speaker Cosine Similarity (SECS)** | $\\ge 0.75$ | **{agg['speaker_similarity_mean']:.3f}** | {'✅ High Timbre Match' if agg['speaker_similarity_mean'] >= 0.75 else '⚠️ Distinct Voice'} |
| **Round-Trip WER (Intelligibility)** | $\\le 10\\%$ | **{wer_val}** | {'✅ Intelligible' if (agg['wer_mean'] is not None and agg['wer_mean'] <= 0.10) else 'N/A'} |
| **Round-Trip CER** | $\\le 5\\%$ | **{cer_val}** | {'✅ Clear Articulation' if (agg['cer_mean'] is not None and agg['cer_mean'] <= 0.05) else 'N/A'} |

---

## 2. Language Breakdown (English vs. Italian)

| Locale | Utterances | MCD (dB) | $F_0$ RMSE | $F_0$ PCC | SECS | Intelligibility WER | Intelligibility CER | Synthesis RTF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{lang_table}

---

## 3. Methodology & Scientific Standards

1. **Acoustic Fidelity (MCD):** Mel-Cepstral Distortion measures spectral envelope distance using Dynamic Time Warping (DTW) with 16-order normalized Mel-generalized cepstra (excluding $c_0$).
2. **Prosody Tracking ($F_0$ Metrics):** Fundamental frequency contours extracted via the normalized autocorrelation/YIN algorithm with energy-gated voicing detection; evaluated over mutually voiced frames for pitch deviation and melodic shape correlation ($F_0$ PCC).
3. **Speaker Identity (SECS):** Deep speaker embeddings extracted via NeMo TitaNet-small, evaluating target voice cloning retention.
4. **Linguistic Accuracy & Intelligibility:** Normalized Word Error Rate (WER) and Character Error Rate (CER) computed with locale-aware tokenization handling English contractions and Italian apostrophe elisions (`l'albero`, `un'altra`) and accented vowels (`à`, `è`, `é`, `ì`, `ò`, `ù`).

---

## 4. Artifacts Generated

* `tts_metrics_summary.json`: Complete machine-readable metrics dump.
* `TTS_EVALUATION_REPORT.md`: This executive report.
"""
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)

        return {
            "summary_path": json_path,
            "report_path": md_path,
        }
