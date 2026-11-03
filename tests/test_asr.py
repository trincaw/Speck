"""Tests for Nemotron 3.5 ASR implementation."""

import pytest
from speck import NemotronASR, LOOKAHEAD_CONFIGS, SUPPORTED_LANGUAGES
from speck.asr.languages import parse_language_tag


def test_language_metadata():
    total_locales = sum(len(tier.locales) for tier in SUPPORTED_LANGUAGES.values())
    assert total_locales == 40
    assert "en-US" in SUPPORTED_LANGUAGES["transcription_ready"].locales
    assert "es-ES" in SUPPORTED_LANGUAGES["transcription_ready"].locales
    assert "it-IT" in SUPPORTED_LANGUAGES["transcription_ready"].locales


def test_parse_language_tag():
    text_with_tag = "Hello world.<en-US>"
    cleaned, tag = parse_language_tag(text_with_tag)
    assert tag == "en-US"
    assert cleaned == "Hello world."

    clean_text = "Good morning."
    cleaned2, tag2 = parse_language_tag(clean_text)
    assert tag2 is None
    assert cleaned2 == "Good morning."


def test_lookahead_configs():
    assert 0 in LOOKAHEAD_CONFIGS
    assert 3 in LOOKAHEAD_CONFIGS
    assert 6 in LOOKAHEAD_CONFIGS
    assert 13 in LOOKAHEAD_CONFIGS
    assert LOOKAHEAD_CONFIGS[0]["latency_ms"] == 80.0
    assert LOOKAHEAD_CONFIGS[6]["latency_ms"] == 560.0


@pytest.fixture(scope="module")
def asr_engine():
    engine = NemotronASR()
    engine.load()
    return engine


def test_transcribe_offline(asr_engine):
    res = asr_engine.transcribe("samples/bcn_weather.mp3", language="en-US")
    assert res.text != ""
    assert "Barcelona" in res.text
    assert res.duration_seconds > 0
    assert res.inference_seconds > 0
    assert res.real_time_factor > 0


def test_transcribe_streaming(asr_engine):
    chunks = list(
        asr_engine.stream_transcribe(
            "samples/bcn_weather.mp3",
            language="en-US",
            lookahead=6,
            skip_special_tokens=True,
        )
    )
    full_text = "".join(chunks).strip()
    assert full_text != ""
    assert "Barcelona" in full_text


def test_live_stream_session(asr_engine):
    import time
    from transformers.audio_utils import load_audio
    from speck.streaming.sessions import LiveAudioStreamSession

    audio = load_audio("samples/bcn_weather.mp3", sampling_rate=16000)
    session = LiveAudioStreamSession(asr_engine, language="en-US", lookahead=6)
    session.start()

    chunk_size = 1600  # 100ms at 16kHz
    for i in range(0, len(audio), chunk_size):
        session.push_audio(audio[i : i + chunk_size])

    session.finish()
    tokens = list(session.iter_tokens())
    text = "".join(tokens).strip()
    assert text != ""
    assert "Barcelona" in text
