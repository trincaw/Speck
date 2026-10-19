"""Tests for live streaming buffers and online speaker classification."""

import numpy as np
from transformers.audio_utils import load_audio

from spek.speakers import SpeakerClassifier, SpeakerEmbedder, VadSegmenter
from spek.streamer import AudioBuffer


def test_audio_buffer_absolute_indexing_and_discard():
    buf = AudioBuffer(capacity=4)
    buf.append(np.arange(10, dtype=np.float32))
    assert len(buf) == 10
    np.testing.assert_array_equal(buf.slice(2, 5), [2, 3, 4])
    buf.discard_before(6)
    buf.append(np.array([10, 11], dtype=np.float32))
    assert len(buf) == 12
    np.testing.assert_array_equal(buf.slice(6, 20), [6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(buf.slice(0, 7), [6])  # discarded samples are gone


class _FakeEmbedder:
    sample_rate = 16000

    def embed(self, samples):
        v = np.asarray(samples[:4], dtype=np.float32)
        return v / np.linalg.norm(v)


def test_classifier_creates_matches_and_enrolls():
    clf = SpeakerClassifier(_FakeEmbedder(), threshold=0.9)
    a = np.r_[1, 0, 0, 0, np.zeros(32000)].astype(np.float32)
    b = np.r_[0, 1, 0, 0, np.zeros(32000)].astype(np.float32)

    assert clf.classify(a).label == "Speaker 0"
    assert clf.classify(b).label == "Speaker 1"
    assert clf.classify(a).label == "Speaker 0"

    clf.enroll("Alice", np.r_[0, 0, 1, 0].astype(np.float32))
    assert clf.classify(np.r_[0, 0, 1, 0.1, np.zeros(32000)].astype(np.float32)).label == "Alice"

    # Short utterances never spawn new speakers
    short = np.r_[0, 0, 0, 1, np.zeros(100)].astype(np.float32)
    assert not clf.classify(short).is_new


def test_vad_and_embeddings_separate_two_real_speakers():
    """Obama vs. weather presenter: the online classifier should tell them apart."""
    sr = 16000
    obama = load_audio("samples/obama.mp3", sampling_rate=sr)[: 30 * sr]
    weather = load_audio("samples/bcn_weather.mp3", sampling_rate=sr)
    # Level-match the clips (the weather recording is ~12x quieter), as one mic would be
    obama, weather = (0.5 * x / np.abs(x).max() for x in (obama, weather))
    audio = np.concatenate([obama, np.zeros(sr), weather]).astype(np.float32)

    vad = VadSegmenter(sr)
    clf = SpeakerClassifier(SpeakerEmbedder())
    labels = []
    for i in range(0, len(audio), 1600):
        labels += [(u.start, clf.classify(u.samples).label) for u in vad.accept(audio[i : i + 1600])]
    labels += [(u.start, clf.classify(u.samples).label) for u in vad.flush()]

    first = {lbl for start, lbl in labels if start < 30}
    second = {lbl for start, lbl in labels if start > 31}
    assert len(labels) >= 3
    assert first and second and first.isdisjoint(second)
