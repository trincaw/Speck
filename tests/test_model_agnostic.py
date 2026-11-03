"""Tests for model-agnostic architecture, config loading and the engine registry."""

import numpy as np
import pytest

from speck.asr import ENGINE_SPECS, BaseASREngine, get_engine, get_spec
from speck.config import load_env_file
from speck.asr.registry import _ENGINE_REGISTRY, register_engine


def test_env_loader_parses_properly(tmp_path, monkeypatch):
    for key in ("SPECK_MODEL", "SPECK_HOTWORDS", "SPECK_LANGUAGE", "SERVER_PORT"):
        monkeypatch.setenv(key, "")  # restored after the test, so the override below doesn't leak
    env_content = """
    # Comment line
    SPECK_MODEL=whisper
    SPECK_HOTWORDS="Avvale, Kubernetes"
    SPECK_LANGUAGE='en'
    export SERVER_PORT=9000
    EMPTY_VAL=
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)

    loaded = load_env_file(filepath=env_file, override=True)
    assert loaded["SPECK_MODEL"] == "whisper"
    assert loaded["SPECK_HOTWORDS"] == "Avvale, Kubernetes"
    assert loaded["SPECK_LANGUAGE"] == "en"
    assert loaded["SERVER_PORT"] == "9000"


def test_builtin_engines_are_parakeet_and_whisper():
    assert [spec.id for spec in ENGINE_SPECS] == ["parakeet", "whisper"]


def test_default_and_unknown_models_fall_back_to_parakeet(monkeypatch):
    from speck.asr.engines.parakeet import ParakeetASR

    monkeypatch.delenv("SPECK_MODEL", raising=False)
    monkeypatch.delenv("ASR_MODEL", raising=False)
    assert isinstance(get_engine(), ParakeetASR)
    assert isinstance(get_engine("nemotron"), ParakeetASR)
    assert get_spec("voxtral").id == "parakeet"


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


def test_factory_resolves_whisper_and_parakeet():
    from speck.asr.engines.parakeet import ParakeetASR
    from speck.asr.engines.whisper import WhisperASR

    for alias, model_id in [("whisper-large-v3-turbo", "large-v3-turbo"), ("whisper", "large-v3-turbo")]:
        engine = get_engine(alias)
        assert isinstance(engine, WhisperASR) and engine.model_id == model_id
    assert get_engine("Systran/faster-whisper-small").model_id == "Systran/faster-whisper-small"
    assert isinstance(get_engine("parakeet"), ParakeetASR)


def test_correct_vocabulary_snaps_near_misses_only():
    from speck.asr.postprocess import correct_vocabulary

    assert correct_vocabulary("Call Avale support", ["Avvale"]) == "Call Avvale support"
    assert correct_vocabulary("deploy on Kubernetis", ["Kubernetes"]) == "deploy on Kubernetes"
    assert correct_vocabulary("the cat sat", ["Kat"]) == "the cat sat"  # too short to rewrite
    assert correct_vocabulary("hello world", []) == "hello world"


def test_rover_majority_vote():
    from speck.asr.postprocess import rover

    votes = ["the cat sat on a mat", "the cat sat on the mat", "the bat sat on the mat today", "the cat sat on the mat"]
    assert rover(votes) == "the cat sat on the mat"
    assert rover(["a b c", "a c", "a b c d"]) == "a b c"
    assert rover([]) == ""


def test_normalize_audio_scales_to_peak_and_spares_silence():
    from speck.audio import normalize_audio

    quiet = (0.2 * np.sin(np.linspace(0, 200, 16000)) + 0.01).astype(np.float32)
    loud = normalize_audio(quiet)
    assert abs(np.abs(loud).max() - 0.9) < 1e-3
    assert abs(loud.mean()) < 1e-3  # DC offset removed
    silent = np.full(1600, 1e-6, dtype=np.float32)
    assert np.abs(normalize_audio(silent)).max() < 1e-4  # no noise amplification


def test_speaker_model_registry(monkeypatch):
    from speck.hub import default_assets, embedding_model

    monkeypatch.delenv("SPECK_SPEAKER_MODEL", raising=False)
    model = embedding_model()
    assert model.id == "titanet-small" and model.match_threshold == 0.35 and model.cluster_threshold == 0.7
    assert [a.id for a in default_assets()] == ["titanet-small", "pyannote-3.0", "silero"]
    with pytest.raises(ValueError, match="Available: titanet-small"):
        embedding_model("wespeaker-resnet34")
