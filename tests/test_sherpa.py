"""Tests for Sherpa-ONNX Speech Tagging & Speaker Diarization Engine."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from spek.asr import BaseASREngine, SherpaASR, SherpaOnnxASR, get_engine
from spek.config import (
    get_sherpa_num_clusters,
    get_sherpa_tag_format,
    get_sherpa_threshold,
)
from spek.engines.sherpa import (
    SherpaSpeakerTagger,
    format_speaker_tagged_transcript,
    merge_speaker_segments,
)
from spek.factory import list_available_models
from spek.server import app, set_engine


def test_factory_resolves_sherpa_aliases():
    """Verify factory returns SherpaOnnxASR for all supported aliases."""
    for alias in ["sherpa", "sherpa-onnx", "sherpa_onnx", "sherpa-asr", "sherpa-onnx-asr"]:
        engine = get_engine(alias)
        assert isinstance(engine, SherpaOnnxASR)
        assert "Sherpa-ONNX" in engine.name
        assert engine.sampling_rate == 16000


def test_list_available_models_includes_sherpa():
    """Verify list_available_models includes Sherpa-ONNX metadata."""
    models = list_available_models()
    sherpa_meta = next((m for m in models if m["id"] == "sherpa-onnx"), None)
    assert sherpa_meta is not None
    assert "Diarization" in sherpa_meta["name"]
    assert sherpa_meta["type"] == "local_onnx"


def test_merge_speaker_segments():
    """Verify contiguous speaker segments within max_gap are properly merged."""
    segments = [
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 0.0, "end": 2.0, "text": "Hello world"},
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 2.3, "end": 4.0, "text": "how are you?"},
        {"speaker": "Speaker 1", "speaker_id": 1, "start": 4.5, "end": 6.0, "text": "I am fine."},
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 6.2, "end": 8.0, "text": "Great to hear."},
    ]
    merged = merge_speaker_segments(segments, max_gap=0.8)
    assert len(merged) == 3
    assert merged[0]["speaker"] == "Speaker 0"
    assert merged[0]["text"] == "Hello world how are you?"
    assert merged[0]["end"] == 4.0
    assert merged[1]["speaker"] == "Speaker 1"
    assert merged[1]["text"] == "I am fine."
    assert merged[2]["speaker"] == "Speaker 0"
    assert merged[2]["text"] == "Great to hear."


def test_format_speaker_tagged_transcript_block_and_inline():
    """Verify speaker tag formatting in block and inline styles."""
    segments = [
        {"speaker": "Speaker 0", "text": "First speaker utterance."},
        {"speaker": "Speaker 1", "text": "Second speaker response."},
    ]

    block_text = format_speaker_tagged_transcript(segments, format_style="block")
    expected_block = "[Speaker 0]: First speaker utterance.\n[Speaker 1]: Second speaker response."
    assert block_text == expected_block

    inline_text = format_speaker_tagged_transcript(segments, format_style="inline")
    expected_inline = "[Speaker 0] First speaker utterance. [Speaker 1] Second speaker response."
    assert inline_text == expected_inline


def test_sherpa_config_helpers(monkeypatch):
    """Verify environment variable parsing for Sherpa configuration."""
    monkeypatch.setenv("SPEK_SHERPA_NUM_CLUSTERS", "3")
    monkeypatch.setenv("SPEK_SHERPA_THRESHOLD", "0.65")
    monkeypatch.setenv("SPEK_SHERPA_TAG_FORMAT", "inline")

    assert get_sherpa_num_clusters() == 3
    assert get_sherpa_threshold() == 0.65
    assert get_sherpa_tag_format() == "inline"


def test_sherpa_transcribe_mocked_diarization_and_asr():
    """Test full transcribe pipeline with mocked diarization and ASR components."""
    engine = SherpaOnnxASR(model_id="sherpa-onnx")
    engine._is_loaded = True

    # Mock diarizer tagger
    mock_tagger = MagicMock(spec=SherpaSpeakerTagger)
    mock_tagger.diarize.return_value = [
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 0.5, "end": 2.0, "duration": 1.5, "confidence": 0.95},
        {"speaker": "Speaker 1", "speaker_id": 1, "start": 2.2, "end": 4.0, "duration": 1.8, "confidence": 0.92},
    ]
    engine._tagger = mock_tagger

    engine._transcribe_batch = MagicMock(return_value=["Speech from speaker 0", "Speech from speaker 1"])

    dummy_audio = np.zeros(64000, dtype=np.float32)  # 4.0s of audio at 16kHz
    result = engine.transcribe(dummy_audio, tag_speakers=True, tag_format="block")

    assert "[Speaker 0]: Speech from speaker 0" in result.text
    assert "[Speaker 1]: Speech from speaker 1" in result.text
    assert result.raw_text == "Speech from speaker 0 Speech from speaker 1"
    assert len(result.segments) == 2
    assert result.segments[0]["speaker"] == "Speaker 0"
    assert result.segments[1]["speaker"] == "Speaker 1"
    assert result.duration_seconds == 4.0


def test_sherpa_transcribe_without_speaker_tags():
    """Verify that when tag_speakers=False, tags are omitted from result.text."""
    engine = SherpaOnnxASR(model_id="sherpa-onnx")
    engine._is_loaded = True

    mock_tagger = MagicMock(spec=SherpaSpeakerTagger)
    mock_tagger.diarize.return_value = [
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 0.0, "end": 2.0, "duration": 2.0, "confidence": 0.9},
    ]
    engine._tagger = mock_tagger
    engine._transcribe_batch = MagicMock(return_value=["Plain transcript without tags"])

    dummy_audio = np.zeros(32000, dtype=np.float32)
    result = engine.transcribe(dummy_audio, tag_speakers=False)

    assert "[Speaker" not in result.text
    assert result.text == "Plain transcript without tags"


def test_sherpa_stream_transcribe():
    """Verify progressive streaming yields speaker-tagged segment lines."""
    engine = SherpaOnnxASR(model_id="sherpa-onnx")
    engine._is_loaded = True

    mock_tagger = MagicMock(spec=SherpaSpeakerTagger)
    mock_tagger.diarize.return_value = [
        {"speaker": "Speaker 0", "speaker_id": 0, "start": 0.0, "end": 1.0, "duration": 1.0, "confidence": 0.9},
        {"speaker": "Speaker 1", "speaker_id": 1, "start": 1.2, "end": 2.5, "duration": 1.3, "confidence": 0.85},
    ]
    engine._tagger = mock_tagger
    engine._transcribe_batch = MagicMock(return_value=["Chunk one", "Chunk two"])

    dummy_audio = np.zeros(40000, dtype=np.float32)
    chunks = list(engine.stream_transcribe(dummy_audio, simulate_realtime=False, tag_speakers=True))

    combined = "".join(chunks)
    assert "[Speaker 0]: Chunk one" in combined
    assert "[Speaker 1]: Chunk two" in combined


def test_server_with_sherpa_engine():
    """Verify FastAPI endpoints (/health, /models, /) when SherpaOnnxASR is active."""
    client = TestClient(app)

    sherpa_engine = SherpaOnnxASR(model_id="sherpa-onnx")
    set_engine(sherpa_engine)

    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.json()
    assert "Sherpa-ONNX" in health_data["engine_name"]
    assert health_data["model_id"] == "sherpa-onnx"

    res_models = client.get("/models")
    assert res_models.status_code == 200
    models_data = res_models.json()
    assert models_data["active_model"] == "sherpa-onnx"

    res_ui = client.get("/")
    assert res_ui.status_code == 200
    assert "Sherpa-ONNX" in res_ui.text


def test_sherpa_live_session_splits_turns_and_reports_speakers():
    """A hand-over without a pause yields two tagged turns plus live speaker events."""
    from transformers.audio_utils import load_audio

    sr = 16000
    obama = load_audio("samples/obama.mp3", sampling_rate=sr)[5 * sr : 11 * sr]
    other = load_audio("samples/sample1.flac", sampling_rate=sr)[: 6 * sr]
    obama, other = (0.5 * x / np.abs(x).max() for x in (obama, other))
    audio = np.concatenate([obama, np.zeros(sr // 10), other]).astype(np.float32)

    engine = SherpaOnnxASR(model_id="sherpa-onnx")
    engine._is_loaded = True
    engine._transcribe_batch = MagicMock(side_effect=lambda chunks: [f"words{len(c) // sr}" for c in chunks])
    events = []
    session = engine.create_stream_session(on_speaker=events.append).start()
    for i in range(0, len(audio), sr // 10):
        session.push_audio(audio[i : i + sr // 10])
    session.finish()
    assert session.join(timeout=60)

    speakers = [s["speaker"] for s in session.segments]
    assert len(set(speakers)) == 2
    assert session.transcript.count("[Speaker") >= 2
    assert any(e["provisional"] for e in events)
    assert [e["label"] for e in events if not e["provisional"]] == speakers
