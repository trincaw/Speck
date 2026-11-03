"""Text-to-Speech (TTS) evaluation pipeline.

Scores synthesized speech against a reference on acoustic quality (MCD), prosody (F0 RMSE / PCC,
VDE, GPE), speaker identity (SECS), round-trip intelligibility (WER, CER, MER, WIL) and synthesis
RTF, then exports a JSON summary and a Markdown report. The metrics themselves live in
`speck.evaluation.acoustic` and `speck.evaluation.text`.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from speck.evaluation.acoustic import AudioLike, compute_f0_metrics, compute_mcd, compute_speaker_similarity, load_audio
from speck.evaluation.text import compute_speech_text_metrics


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
        ref, syn = load_audio(ref_audio, sr), load_audio(syn_audio, sr)  # decode files once for all metrics
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


def synthetic_benchmark(sample_rate: int = 16000) -> List[Dict[str, Any]]:
    """Two built-in reference / "synthesized" tone pairs (en-US, it-IT): a smoke test of the pipeline,
    not a measurement (the ASR text equals the prompt, so WER / CER are 0)."""
    t = np.linspace(0, 1.5, int(sample_rate * 1.5), endpoint=False, dtype=np.float32)

    def tone(f0: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
        ref = (0.5 * np.sin(2 * np.pi * f0 * t) + 0.2 * np.sin(2 * np.pi * 2 * f0 * t)).astype(np.float32)
        return ref, ref * 1.02 + (0.0005 * np.random.RandomState(seed).randn(len(t))).astype(np.float32)

    items = [  # (language, f0 Hz, noise seed, synthesis time s, text)
        ("en-US", 140, 1, 0.12, "The quick brown fox jumps over the lazy dog."),
        ("it-IT", 190, 2, 0.13, "L'intelligenza artificiale per l'elaborazione del linguaggio ed il parlato."),
    ]
    benchmarks = []
    for language, f0, seed, synthesis_time, text in items:
        ref, syn = tone(f0, seed)
        benchmarks.append({"ref_audio": ref, "syn_audio": syn, "text": text, "asr_text": text,
                           "language": language, "synthesis_time_sec": synthesis_time})
    return benchmarks
