"""Tests for live streaming buffers and online speaker classification."""

import numpy as np
from transformers.audio_utils import load_audio

from spek.speakers import LiveSpeakerPreview, SpeakerClassifier, SpeakerEmbedder, VadSegmenter
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


def _two_voices(gap: float):
    sr = 16000
    obama = load_audio("samples/obama.mp3", sampling_rate=sr)[5 * sr : 11 * sr]
    other = load_audio("samples/sample1.flac", sampling_rate=sr)[: 6 * sr]
    obama, other = (0.5 * x / np.abs(x).max() for x in (obama, other))
    return np.concatenate([obama, np.zeros(int(gap * sr)), other]).astype(np.float32), sr


def test_classifier_merges_duplicate_speakers_and_resolves_labels():
    clf = SpeakerClassifier(_FakeEmbedder(), threshold=0.9, merge_threshold=0.85)
    pad = np.zeros(32000)
    clf.classify(np.r_[1, 0, 0, 0, pad].astype(np.float32))
    clf.classify(np.r_[1, 0.8, 0, 0, pad].astype(np.float32))  # cos 0.78: becomes "Speaker 1"
    assert len(clf.speakers) == 2
    # A voice in between pulls Speaker 1 towards Speaker 0, and they are folded together
    assert clf.classify(np.r_[1, 0.4, 0, 0, pad].astype(np.float32)).label == "Speaker 0"
    assert [s.label for s in clf.speakers] == ["Speaker 0"]
    assert clf.resolve("Speaker 1") == "Speaker 0"
    # Numbering never reuses a merged label
    assert clf.classify(np.r_[0, 0, 0, 1, pad].astype(np.float32)).label == "Speaker 2"


def test_classifier_peek_does_not_learn():
    clf = SpeakerClassifier(_FakeEmbedder(), threshold=0.9)
    a = np.r_[1, 0, 0, 0, np.zeros(32000)].astype(np.float32)
    assert clf.peek(a).label == "?"
    assert not clf.speakers
    clf.classify(a)
    before = clf.speakers[0].centroid.copy()
    assert clf.peek(a).label == "Speaker 0"
    np.testing.assert_array_equal(clf.speakers[0].centroid, before)


def test_segment_splits_handover_without_pause():
    """Two voices 0.1 s apart form one VAD utterance; segment() must still split them."""
    audio, sr = _two_voices(gap=0.1)
    clf = SpeakerClassifier(SpeakerEmbedder())
    turns = clf.segment(audio)
    assert len(turns) == 2
    assert turns[0].match.label != turns[1].match.label
    assert abs(turns[0].end - 6.0) < 0.75


def test_segment_keeps_single_speaker_whole():
    sr = 16000
    obama = load_audio("samples/obama.mp3", sampling_rate=sr)[5 * sr : 17 * sr].astype(np.float32)
    turns = SpeakerClassifier(SpeakerEmbedder()).segment(obama)
    assert len(turns) == 1 and turns[0].end == len(obama) / sr


def test_live_preview_labels_speech_in_progress():
    audio, sr = _two_voices(gap=1.0)
    vad = VadSegmenter(sr)
    clf = SpeakerClassifier(SpeakerEmbedder())
    preview = LiveSpeakerPreview(clf, vad)
    live = []
    for i in range(0, len(audio), 1600):
        for utt in vad.accept(audio[i : i + 1600]):
            clf.segment(utt.samples)
        if match := preview.update():
            live.append((i / sr, match.label))
    assert vad.live_seconds > 0 and len(vad.live_audio(2.0)) <= 2 * sr + 512
    # Unknown before its first utterance ends, then recognised live; the second voice is new
    assert live and all(lbl == "?" for t, lbl in live if t < 3.0)
    assert {lbl for t, lbl in live if 4.5 < t < 6.0} == {"Speaker 0"}
    assert {lbl for t, lbl in live if t > 8.0} == {"?"}
