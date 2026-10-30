"""Tests for model-agnostic architecture, config loading, and Voxtral Mini Transcribe 2."""

import io
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from speck.asr import BaseASREngine, NemotronASR, VoxtralASR, get_engine
from speck.config import load_env_file
from speck.factory import _ENGINE_REGISTRY, list_available_models, register_engine


def test_env_loader_parses_properly(tmp_path):
    env_content = """
    # Comment line
    SPECK_MODEL=voxtral-mini-transcribe-2
    MISTRAL_API_KEY="test-secret-key-12345"
    VOXTRAL_BACKEND='api'
    export SERVER_PORT=9000
    EMPTY_VAL=
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    loaded = load_env_file(filepath=env_file, override=True)
    assert loaded["SPECK_MODEL"] == "voxtral-mini-transcribe-2"
    assert loaded["MISTRAL_API_KEY"] == "test-secret-key-12345"
    assert loaded["VOXTRAL_BACKEND"] == "api"
    assert loaded["SERVER_PORT"] == "9000"


def test_factory_resolves_nemotron():
    engine = get_engine("nemotron")
    assert isinstance(engine, NemotronASR)
    assert "nvidia/nemotron-3.5-asr-streaming-0.6b" in engine.model_id
    assert engine.sampling_rate == 16000


def test_factory_resolves_voxtral_aliases():
    for alias in ["voxtral", "voxtral-mini", "voxtral-mini-transcribe-2", "voxtral-mini-latest"]:
        engine = get_engine(alias)
        assert isinstance(engine, VoxtralASR)
        assert "Voxtral" in engine.name
        assert engine.sampling_rate == 16000


def test_factory_custom_engine_registration():
    class DummyEngine(BaseASREngine):
        @property
        def name(self):
            return "Dummy"

        @property
        def sampling_rate(self):
            return 8000

        def load(self):
            return self

        def transcribe(self, audio, **kwargs):
            pass

        def stream_transcribe(self, audio, **kwargs):
            yield ""

    register_engine("dummy-asr", DummyEngine)
    assert "dummy-asr" in _ENGINE_REGISTRY
    engine = get_engine("dummy-asr")
    assert isinstance(engine, DummyEngine)
    assert engine.sampling_rate == 8000


def test_voxtral_requires_api_key_in_api_mode(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    engine = VoxtralASR(backend="api", api_key="")
    engine.load()

    dummy_audio = np.zeros(16000, dtype=np.float32)
    with pytest.raises(ValueError, match="Mistral API key is required"):
        engine.transcribe(dummy_audio)


def test_voxtral_transcribe_mocked_api():
    engine = VoxtralASR(
        model_id="voxtral-mini-latest",
        backend="api",
        api_key="mock-api-key",
    )
    engine.load()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Hello world from Voxtral.",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello world"},
            {"start": 1.0, "end": 2.0, "text": "from Voxtral."},
        ],
    }

    dummy_audio = np.zeros(32000, dtype=np.float32)  # 2.0s of audio at 16kHz

    with patch("requests.Session.post", return_value=mock_response) as mock_post:
        result = engine.transcribe(dummy_audio, language="en-US")

        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer mock-api-key"
        assert kwargs["data"]["model"] == "voxtral-mini-latest"
        assert kwargs["data"]["language"] == "en"

        assert result.text == "Hello world from Voxtral."
        assert result.detected_language == "en"
        assert result.duration_seconds == 2.0
        assert len(result.segments) == 2


def test_voxtral_stream_transcribe():
    engine = VoxtralASR(
        model_id="voxtral-mini-latest",
        backend="api",
        api_key="mock-api-key",
    )
    engine.load()

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "text": "Streaming test from Voxtral",
        "segments": [
            {"start": 0.0, "end": 0.5, "text": "Streaming test"},
            {"start": 0.5, "end": 1.0, "text": "from Voxtral"},
        ],
    }

    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch("requests.Session.post", return_value=mock_response):
        chunks = list(engine.stream_transcribe(dummy_audio, simulate_realtime=False))
        text = "".join(chunks).strip()
        assert "Streaming test" in text
        assert "from Voxtral" in text


def test_factory_resolves_whisper_and_parakeet():
    from speck.engines.parakeet import ParakeetASR
    from speck.engines.whisper import WhisperASR

    for alias, model_id in [("whisper-base", "base"), ("whisper-large-v3-turbo", "large-v3-turbo"), ("whisper", "large-v3-turbo")]:
        engine = get_engine(alias)
        assert isinstance(engine, WhisperASR) and engine.model_id == model_id
    assert get_engine("Systran/faster-whisper-small").model_id == "Systran/faster-whisper-small"
    assert isinstance(get_engine("parakeet"), ParakeetASR)


def test_correct_vocabulary_snaps_near_misses_only():
    from speck.postprocess import correct_vocabulary

    assert correct_vocabulary("Call Avale support", ["Avvale"]) == "Call Avvale support"
    assert correct_vocabulary("deploy on Kubernetis", ["Kubernetes"]) == "deploy on Kubernetes"
    assert correct_vocabulary("the cat sat", ["Kat"]) == "the cat sat"  # too short to rewrite
    assert correct_vocabulary("hello world", []) == "hello world"


def test_rover_majority_vote():
    from speck.postprocess import rover

    votes = ["the cat sat on a mat", "the cat sat on the mat", "the bat sat on the mat today", "the cat sat on the mat"]
    assert rover(votes) == "the cat sat on the mat"
    assert rover(["a b c", "a c", "a b c d"]) == "a b c"
    assert rover([]) == ""


def test_normalize_audio_scales_to_peak_and_spares_silence():
    from speck.utils import normalize_audio

    quiet = (0.2 * np.sin(np.linspace(0, 200, 16000)) + 0.01).astype(np.float32)
    loud = normalize_audio(quiet)
    assert abs(np.abs(loud).max() - 0.9) < 1e-3
    assert abs(loud.mean()) < 1e-3  # DC offset removed
    silent = np.full(1600, 1e-6, dtype=np.float32)
    assert np.abs(normalize_audio(silent)).max() < 1e-4  # no noise amplification
