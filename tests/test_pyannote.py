import numpy as np
import pytest
from transformers.audio_utils import load_audio

from speck.pyannote import MATCH_THRESHOLD, PyannoteSpeakerRecognizer, RealTimeStreamingEngine


@pytest.fixture(scope="module")
def recognizer():
    return PyannoteSpeakerRecognizer()


def test_embedding_extraction_and_normalization(recognizer):
    audio = np.random.randn(16000 * 2).astype(np.float32)
    emb, elapsed = recognizer.extract_embedding(audio)
    assert emb.ndim == 1
    assert emb.shape[0] > 0
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4
    assert elapsed > 0.0


def test_enrollment_and_matching(recognizer):
    rec = PyannoteSpeakerRecognizer()
    spk1_audio = np.random.randn(16000 * 2).astype(np.float32)
    spk2_audio = np.random.randn(16000 * 2).astype(np.float32)

    rec.enroll("alice", spk1_audio)
    rec.enroll("bob", spk2_audio)

    assert "alice" in rec.database
    assert "bob" in rec.database

    scores, elapsed = rec.match(spk1_audio)
    assert scores["alice"] > scores["bob"]
    assert elapsed > 0.0


def test_verification_separation(recognizer):
    sr = 16000
    obama1 = load_audio("samples/obama.mp3", sampling_rate=sr)[: 3 * sr]
    obama2 = load_audio("samples/obama.mp3", sampling_rate=sr)[5 * sr : 8 * sr]
    sample1 = load_audio("samples/sample1.flac", sampling_rate=sr)[: 3 * sr]

    same_score, _ = recognizer.verify(obama1, obama2)
    diff_score, _ = recognizer.verify(obama1, sample1)

    assert same_score > MATCH_THRESHOLD > diff_score


def test_diarize_two_speakers(recognizer):
    sr = 16000
    obama = load_audio("samples/obama.mp3", sampling_rate=sr)[5 * sr : 10 * sr]
    sample1 = load_audio("samples/sample1.flac", sampling_rate=sr)[: 5 * sr]
    obama = 0.5 * obama / np.abs(obama).max()
    sample1 = 0.5 * sample1 / np.abs(sample1).max()
    combined = np.concatenate([obama, np.zeros(int(0.2 * sr)), sample1]).astype(np.float32)

    turns = recognizer.diarize(combined, num_speakers=2)
    assert len(turns) >= 2
    assert turns[0].start < 5.0
    assert turns[-1].end > 5.0


def test_streaming_compatibility(recognizer):
    rec = PyannoteSpeakerRecognizer()
    audio = np.random.randn(16000 * 2).astype(np.float32)
    rec.enroll("alice", audio)

    engine = RealTimeStreamingEngine(rec, chunk_sec=0.5, window_sec=1.5, threshold=0.7)
    chunk = audio[:8000]
    spk, score, lat, rtf = engine.process_chunk(chunk)
    assert isinstance(spk, str)
    assert isinstance(score, float)
    assert lat > 0.0
    assert rtf > 0.0
