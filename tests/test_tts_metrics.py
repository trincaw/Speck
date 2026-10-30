"""Tests for Text-to-Speech (TTS) and linguistic speech evaluation metrics.

Validates:
1. Text normalization for English (contractions, punctuation, whitespace)
2. Text normalization for Italian (apostrophe elisions, accented vowels, apocope)
3. Word Error Rate (WER) and Character Error Rate (CER) with Levenshtein alignments
4. Match Error Rate (MER), Word Information Lost (WIL), and Word Information Preserved (WIP)
5. Mel-Cepstral Distortion (MCD) in dB via Dynamic Time Warping (DTW)
6. F0 pitch contour metrics (RMSE in Hz and cents, PCC prosody correlation, VDE, GPE)
7. Speaker Embedding Cosine Similarity (SECS)
8. End-to-end TTSMetricsPipeline and artifact generation for both English and Italian
"""

import json
import os
import numpy as np
import pytest

from speck.metrics import (
    MCD_FACTOR,
    TTSMetricsPipeline,
    compute_cer,
    compute_f0_metrics,
    compute_mcd,
    compute_mer_and_wil,
    compute_speaker_similarity,
    compute_speech_text_metrics,
    compute_wer,
    normalize_english_text,
    normalize_italian_text,
    normalize_text,
)


# ==============================================================================
# 1. TEXT NORMALIZATION TESTS (ENGLISH & ITALIAN)
# ==============================================================================

def test_normalize_english_contractions():
    raw = "I can't believe it's already Monday, won't you join us? We're ready!"
    normalized = normalize_english_text(raw)
    assert "cannot" in normalized
    assert "it is" in normalized
    assert "will not" in normalized
    assert "we are" in normalized
    assert "!" not in normalized
    assert "," not in normalized
    assert "?" not in normalized


def test_normalize_english_punctuation_and_casing():
    raw = "  Hello, WORLD!   Testing TTS text-to-speech normalization... 123  "
    normalized = normalize_english_text(raw)
    assert normalized == "hello world testing tts text to speech normalization 123"


def test_normalize_italian_elisions_with_apostrophe():
    # In Italian, articles and prepositions elide before vowels: l'albero, un'amica, d'accordo
    raw = "L'albero nel giardino è fiorito, ed è un'altra bellissima giornata d'estate!"
    normalized = normalize_italian_text(raw, preserve_accents=True)
    
    # Apostrophe elision should be separated: "l' albero", "un' altra", "d' estate"
    assert "l' albero" in normalized
    assert "un' altra" in normalized
    assert "d' estate" in normalized
    # Accented vowels must be preserved
    assert "è" in normalized


def test_normalize_italian_accents_preservation_and_folding():
    raw = "Perché la città di Roma è così affascinante? Più virtù e caffè per tutti."
    
    # Strict mode: preserve distinct Italian accented vowels (à, è, é, ì, ò, ù)
    norm_strict = normalize_italian_text(raw, preserve_accents=True)
    assert "perché" in norm_strict
    assert "città" in norm_strict
    assert "è" in norm_strict
    assert "più" in norm_strict
    assert "caffè" in norm_strict
    assert "virtù" in norm_strict

    # Folded mode: mapped to plain ASCII vowels
    norm_folded = normalize_italian_text(raw, preserve_accents=False)
    assert "perche" in norm_folded
    assert "citta" in norm_folded
    assert "e" in norm_folded
    assert "piu" in norm_folded
    assert "caffe" in norm_folded
    assert "virtu" in norm_folded


def test_normalize_italian_typography_quotes():
    raw = "«La tecnologia vocale — diceva l'esperto — è pronta per l'uso quotidiano.»"
    normalized = normalize_italian_text(raw)
    assert "«" not in normalized
    assert "»" not in normalized
    assert "—" not in normalized
    assert "l' esperto" in normalized
    assert "l' uso" in normalized


def test_normalize_text_dispatcher():
    en_res = normalize_text("Don't stop", language="en-US")
    assert en_res == "do not stop"

    it_res = normalize_text("L'amico", language="it-IT")
    assert it_res == "l' amico"


# ==============================================================================
# 2. WORD ERROR RATE (WER) & CHARACTER ERROR RATE (CER) TESTS
# ==============================================================================

def test_compute_wer_perfect_match():
    ref = "The quick brown fox jumps over the lazy dog."
    hyp = "The quick brown fox jumps over the lazy dog."
    res = compute_wer(ref, hyp, language="en")
    assert res["wer"] == 0.0
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert res["insertions"] == 0
    assert res["hits"] == 9


def test_compute_wer_with_substitutions_and_insertions():
    # ref: "the brown fox" (3 words)
    # hyp: "the red brown fox jumps" (S=0, D=0, I=2: 'red', 'jumps'; H=3)
    ref = "the brown fox"
    hyp = "the red brown fox jumps"
    res = compute_wer(ref, hyp, language="en", normalize=False)
    assert res["hits"] == 3
    assert res["insertions"] == 2
    assert res["substitutions"] == 0
    assert res["deletions"] == 0
    assert abs(res["wer"] - (2 / 3)) < 1e-6


def test_compute_wer_italian_sentences():
    ref = "L'intelligenza artificiale per l'analisi del linguaggio naturale."
    hyp = "L'intelligenza artificiale per analisi del linguaggio naturale."
    # 'l\'' is deleted in hypothesis
    res = compute_wer(ref, hyp, language="it")
    assert res["deletions"] == 1
    assert res["substitutions"] == 0
    assert res["insertions"] == 0
    assert 0.0 < res["wer"] < 0.20


def test_compute_wer_italian_accent_distinction():
    # In Italian, 'e' (conjunction: and) and 'è' (verb: is) are completely different phonemes and words
    ref = "Roma è la capitale e la città eterna."
    hyp = "Roma e la capitale e la città eterna."  # 'è' mistranscribed as 'e'
    res = compute_wer(ref, hyp, language="it", preserve_accents=True)
    assert res["substitutions"] == 1
    assert res["wer"] > 0.0


def test_compute_cer_english_and_italian():
    ref_en = "synthesis"
    hyp_en = "synthesys"  # 1 substitution ('i' -> 'y')
    cer_en = compute_cer(ref_en, hyp_en, language="en", normalize=False)
    assert cer_en["substitutions"] == 1
    assert abs(cer_en["cer"] - (1 / 9)) < 1e-6

    ref_it = "città"
    hyp_it = "citta"  # accent mismatch
    cer_it = compute_cer(ref_it, hyp_it, language="it", preserve_accents=True)
    assert cer_it["substitutions"] == 1
    assert abs(cer_it["cer"] - (1 / 5)) < 1e-6


def test_compute_mer_and_wil():
    ref = "machine learning and speech synthesis"
    hyp = "machine learning for speech synthesis"  # 1 substitution ('and' -> 'for')
    res = compute_mer_and_wil(ref, hyp, language="en")
    assert 0.0 < res["mer"] < 0.30
    assert 0.0 < res["wil"] < 0.40
    assert 0.60 < res["wip"] <= 1.0


def test_compute_speech_text_metrics_empty_inputs():
    res = compute_speech_text_metrics("", "", language="en")
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0
    assert res["wil"] == 0.0
    assert res["wip"] == 1.0

    res_hyp_empty = compute_speech_text_metrics("Hello world", "", language="en")
    assert res_hyp_empty["wer"] == 1.0
    assert res_hyp_empty["deletions"] == 2


# ==============================================================================
# 3. ACOUSTIC METRICS: MEL-CEPSTRAL DISTORTION (MCD)
# ==============================================================================

@pytest.fixture
def synthetic_speech_signals():
    """Generates synthetic speech-like signals for acoustic benchmarking."""
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    
    # Harmonic carrier (f0 = 150 Hz + harmonics) modulated by a vowel-like formant envelope
    carrier = (
        0.5 * np.sin(2 * np.pi * 150 * t) +
        0.3 * np.sin(2 * np.pi * 300 * t) +
        0.2 * np.sin(2 * np.pi * 450 * t) +
        0.1 * np.sin(2 * np.pi * 900 * t)
    ).astype(np.float32)
    
    # Formant modulation (amplitude envelope)
    envelope = (0.5 * (1 + np.sin(2 * np.pi * 3 * t))).astype(np.float32)
    reference = carrier * envelope

    # Slightly perturbed synthesis (low noise, representing high-quality TTS)
    noise = (0.0005 * np.random.RandomState(42).randn(len(t))).astype(np.float32)
    synthesized_good = reference * 1.05 + noise

    # Severely degraded synthesis (different fundamental and harmonics)
    carrier_bad = (
        0.5 * np.sin(2 * np.pi * 380 * t) +
        0.3 * np.sin(2 * np.pi * 760 * t) +
        0.2 * np.sin(2 * np.pi * 1140 * t)
    ).astype(np.float32)
    synthesized_bad = carrier_bad * envelope

    return {
        "sr": sr,
        "ref": reference,
        "syn_good": synthesized_good,
        "syn_bad": synthesized_bad,
    }


def test_mcd_identical_signals_is_zero(synthetic_speech_signals):
    ref = synthetic_speech_signals["ref"]
    sr = synthetic_speech_signals["sr"]
    mcd = compute_mcd(ref, ref, sample_rate=sr)
    assert mcd == 0.0


def test_mcd_monotonicity_and_scale(synthetic_speech_signals):
    ref = synthetic_speech_signals["ref"]
    syn_good = synthetic_speech_signals["syn_good"]
    syn_bad = synthetic_speech_signals["syn_bad"]
    sr = synthetic_speech_signals["sr"]

    mcd_good = compute_mcd(ref, syn_good, sample_rate=sr)
    mcd_bad = compute_mcd(ref, syn_bad, sample_rate=sr)

    # MCD must be non-negative
    assert mcd_good > 0.0
    assert mcd_bad > 0.0

    # Slight perturbation should yield low distortion (< 5.0 dB)
    assert mcd_good < 5.0

    # Severely divergent spectral signal must have significantly higher MCD
    assert mcd_bad > mcd_good


def test_mcd_on_real_dataset_samples():
    # Evaluate on CMU Arctic sample WAV files in data/
    ref_file = "data/slt/arctic_a0001.wav"
    syn_file = "data/slt/arctic_a0002.wav"

    if os.path.exists(ref_file):
        # Identical file test
        mcd_self = compute_mcd(ref_file, ref_file)
        assert mcd_self == 0.0

        if os.path.exists(syn_file):
            # Cross-utterance test (same speaker, different sentence)
            mcd_cross = compute_mcd(ref_file, syn_file)
            assert 2.0 < mcd_cross < 12.0


# ==============================================================================
# 4. PROSODY & INTONATION: F0 PITCH METRICS
# ==============================================================================

def test_f0_metrics_identical_signal(synthetic_speech_signals):
    ref = synthetic_speech_signals["ref"]
    sr = synthetic_speech_signals["sr"]
    f0_res = compute_f0_metrics(ref, ref, sample_rate=sr)

    assert f0_res["f0_rmse_hz"] == 0.0
    assert f0_res["f0_rmse_cents"] == 0.0
    assert f0_res["f0_pcc"] == 1.0
    assert f0_res["vde"] == 0.0
    assert f0_res["gpe"] == 0.0


def test_f0_metrics_pitch_shift():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Reference tone at 200 Hz
    ref = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    # Synthesized tone shifted by 20 Hz to 220 Hz
    syn = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    f0_res = compute_f0_metrics(ref, syn, sample_rate=sr)
    assert f0_res["mutually_voiced_frames"] > 0
    # RMSE should be approximately 20 Hz (+/- 2 Hz tolerance due to windowing)
    assert abs(f0_res["f0_rmse_hz"] - 20.0) < 3.0
    # Both are constant tones, so VDE should be 0.0
    assert f0_res["vde"] < 0.05


def test_f0_voicing_decision_error():
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)

    # Pure voiced tone
    ref = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    # Complete silence (unvoiced)
    syn = np.zeros_like(ref)

    f0_res = compute_f0_metrics(ref, syn, sample_rate=sr)
    # Since one is voiced and the other is unvoiced, VDE should be high
    assert f0_res["vde"] > 0.80
    assert f0_res["mutually_voiced_frames"] == 0


# ==============================================================================
# 5. SPEAKER SIMILARITY (SECS) TESTS
# ==============================================================================

def test_speaker_similarity_identical_signals(synthetic_speech_signals):
    ref = synthetic_speech_signals["ref"]
    sim = compute_speaker_similarity(ref, ref, sample_rate=synthetic_speech_signals["sr"])
    assert abs(sim - 1.0) < 1e-4


def test_speaker_similarity_real_speakers():
    slt_1 = "data/slt/arctic_a0001.wav"
    slt_2 = "data/slt/arctic_a0002.wav"
    bdl_1 = "data/bdl/arctic_a0001.wav"

    if os.path.exists(slt_1) and os.path.exists(slt_2) and os.path.exists(bdl_1):
        # Same speaker (slt), different sentences
        sim_same = compute_speaker_similarity(slt_1, slt_2)
        # Different speakers (slt female vs bdl male)
        sim_diff = compute_speaker_similarity(slt_1, bdl_1)

        # Same speaker similarity must exceed cross-speaker similarity
        assert sim_same > sim_diff
        assert sim_same > 0.60
        assert sim_diff < 0.50


# ==============================================================================
# 6. TTS METRICS PIPELINE: ENGLISH & ITALIAN BENCHMARKS & REPORT EXPORT
# ==============================================================================

def test_tts_metrics_pipeline_end_to_end(tmp_path):
    output_dir = str(tmp_path / "tts_reports")
    pipeline = TTSMetricsPipeline(sample_rate=16000, output_dir=output_dir)

    # Construct multilingual test dataset
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    
    # Synthetic carrier for English speaker
    en_ref_wav = (0.5 * np.sin(2 * np.pi * 140 * t) + 0.2 * np.sin(2 * np.pi * 280 * t)).astype(np.float32)
    en_syn_wav = en_ref_wav * 1.02 + (0.0005 * np.random.RandomState(1).randn(len(t))).astype(np.float32)

    # Synthetic carrier for Italian speaker
    it_ref_wav = (0.5 * np.sin(2 * np.pi * 190 * t) + 0.2 * np.sin(2 * np.pi * 380 * t)).astype(np.float32)
    it_syn_wav = it_ref_wav * 1.02 + (0.0005 * np.random.RandomState(2).randn(len(t))).astype(np.float32)

    dataset = [
        # English benchmark items
        {
            "ref_audio": en_ref_wav,
            "syn_audio": en_syn_wav,
            "text": "The quick brown fox jumps over the lazy dog.",
            "asr_text": "The quick brown fox jumps over the lazy dog.",
            "language": "en-US",
            "synthesis_time_sec": 0.12,
        },
        {
            "ref_audio": en_ref_wav,
            "syn_audio": en_syn_wav,
            "text": "Deep learning models synthesize high fidelity natural speech.",
            "asr_text": "Deep learning models synthesize high fidelity natural speech.",
            "language": "en-US",
            "synthesis_time_sec": 0.15,
        },
        # Italian benchmark items
        {
            "ref_audio": it_ref_wav,
            "syn_audio": it_syn_wav,
            "text": "La sintesi vocale neurale produce un parlato chiaro e naturale.",
            "asr_text": "La sintesi vocale neurale produce un parlato chiaro e naturale.",
            "language": "it-IT",
            "synthesis_time_sec": 0.14,
        },
        {
            "ref_audio": it_ref_wav,
            "syn_audio": it_syn_wav,
            "text": "L'intelligenza artificiale per l'elaborazione del linguaggio ed il parlato.",
            "asr_text": "L'intelligenza artificiale per l'elaborazione del linguaggio ed il parlato.",
            "language": "it-IT",
            "synthesis_time_sec": 0.13,
        },
    ]

    eval_output = pipeline.evaluate_dataset(dataset)
    artifacts = pipeline.export_reports(eval_output)

    # Assert artifacts are written
    assert os.path.exists(artifacts["summary_path"])
    assert os.path.exists(artifacts["report_path"])

    with open(artifacts["summary_path"], encoding="utf-8") as f:
        summary = json.load(f)
        assert summary["total_samples"] == 4
        assert "en" in summary["languages"]
        assert "it" in summary["languages"]
        assert summary["aggregate"]["mcd_db_mean"] < 5.0
        assert summary["languages"]["en"]["wer_mean"] == 0.0
        assert summary["languages"]["it"]["wer_mean"] == 0.0

    with open(artifacts["report_path"], encoding="utf-8") as f:
        md_text = f.read()
        assert "Text-to-Speech (TTS) & Speech Evaluation Report" in md_text
        assert "English (en-US)" in md_text
        assert "Italian (it-IT)" in md_text
        assert "Mel-Cepstral Distortion (MCD)" in md_text


def test_normalize_english_drops_fillers_and_spells_numbers():
    assert normalize_english_text("It was uh thirty five degrees, minus um twenty.") == "it was 35 degrees minus 20"
    assert normalize_english_text("one hundred and five") == "105"
    assert normalize_english_text("two thousand twenty four") == "2024"
    assert normalize_english_text("one hundred twenty three thousand four hundred and fifty six") == "123456"
    assert normalize_english_text("twenty twenty") == "20 20"  # numbers that do not compose stay apart
    assert normalize_english_text("seven and eight") == "7 and 8"
    assert normalize_english_text("1,000 and 99%") == "1000 and 99 percent"


def test_wer_ignores_fillers_and_number_format():
    result = compute_wer("It was 35 degrees in Barcelona.", "It was uh thirty five degrees in Barcelona.")
    assert result["wer"] == 0.0


def test_normalize_italian_drops_fillers():
    assert normalize_italian_text("ehm buongiorno a tutti") == "buongiorno a tutti"
