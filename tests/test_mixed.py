"""Tests for mixed mode: microphone + system audio as two channels in one transcript."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from transformers.audio_utils import load_audio

from spek.engines.sherpa import SherpaOnnxASR
from spek.mixed import EchoGate, MixedSession, TranscriptMixer
from spek.system_audio import DualCapture

SR = 16000


def _voice(path, start, seconds):
    x = load_audio(path, sampling_rate=SR)[int(start * SR) : int((start + seconds) * SR)].astype(np.float32)
    return 0.5 * x / np.abs(x).max()


def test_mixer_tags_label_changes_and_joins_tokens_and_turns():
    out = []
    mixer = TranscriptMixer(on_token=out.append)
    mixer.emit("You", "Hello")
    mixer.emit("You", " there")  # incremental token keeps its spacing
    mixer.emit("Speaker 0", "Hi.", whole=True)
    mixer.emit("Speaker 0", "How are you?", whole=True)
    mixer.emit("You", " Fine")
    assert mixer.transcript == "[You]: Hello there\n[Speaker 0]: Hi. How are you?\n[You]: Fine"
    assert "".join(out).strip() == mixer.transcript
    assert [s["speaker"] for s in mixer.segments] == ["You", "Speaker 0", "You"]

    inline = TranscriptMixer(tag_format="inline")
    inline.emit("You", "a")
    inline.emit("Remote", "b")
    assert inline.transcript == "[You] a [Remote] b"

    plain = TranscriptMixer(tag=False)
    plain.emit("You", "a")
    plain.emit("Remote", "  b")
    plain.emit("Remote", "   ")  # blank text is ignored
    assert plain.transcript == "a b"


def _blocks(ref, mic, gate):
    muted = speech = 0
    for i in range(0, min(len(ref), len(mic)) - 1600, 1600):
        gate.reference(ref[i : i + 1600])
        out = gate.filter(mic[i : i + 1600])
        if np.sqrt(np.mean(mic[i : i + 1600] ** 2)) > 0.01:
            speech += 1
            muted += not out.any()
    return muted / speech


def test_echo_gate_mutes_speaker_echo_but_not_your_voice():
    remote = _voice("samples/obama.mp3", 5, 12)
    me = _voice("samples/sample1.flac", 0, 12)
    delay = int(0.15 * SR)
    room = np.exp(-np.arange(800) / 200) / 60
    echo = (0.25 * np.convolve(np.r_[np.zeros(delay), remote], room)[: len(remote)]).astype(np.float32)

    assert _blocks(remote, echo, EchoGate(SR)) > 0.9
    assert _blocks(remote, me, EchoGate(SR)) < 0.2
    assert _blocks(np.zeros_like(me), me, EchoGate(SR)) == 0.0
    assert _blocks(remote, (me + echo).astype(np.float32), EchoGate(SR)) < 0.2  # double-talk passes


class _ListReader:
    def __init__(self, blocks):
        self.blocks = list(blocks)

    def read(self):
        return self.blocks.pop(0) if self.blocks else None

    def close(self):
        pass


def test_dual_capture_routes_each_device_to_its_channel():
    got = {"mic": [], "system": []}
    readers = {"m": _ListReader([np.ones(4, np.float32)] * 2), "s": _ListReader([np.zeros(4, np.float32)] * 3)}
    cap = DualCapture(got["mic"].append, got["system"].append, mic_device="m", system_device="s",
                      reader_factory=lambda device, **_: readers[device]).start()
    cap.stop()
    assert len(got["mic"]) == 2 and all(b.all() for b in got["mic"])
    assert len(got["system"]) == 3 and not any(b.any() for b in got["system"])


def _engine():
    engine = SherpaOnnxASR(model_id="sherpa-onnx")
    engine._is_loaded = True
    engine._transcribe_batch = MagicMock(side_effect=lambda chunks: [f"w{len(c) // SR}" for c in chunks])
    return engine


def _run(session, mic, system):
    session.start()
    n = max(len(mic), len(system))
    mic, system = np.pad(mic, (0, n - len(mic))), np.pad(system, (0, n - len(system)))
    for i in range(0, n, SR // 10):
        session.push_system(system[i : i + SR // 10])
        session.push_mic(mic[i : i + SR // 10])
    session.finish()
    assert session.join(timeout=60)


@pytest.mark.parametrize("classify", [True, False])
def test_mixed_session_tags_you_and_remote_speakers(classify):
    """Mic voice is always "You"; two remote voices are told apart only when classifying."""
    gap = np.zeros(SR, np.float32)
    mic = np.concatenate([np.zeros(14 * SR, np.float32), _voice("samples/sample1.flac", 0, 5)])
    system = np.concatenate([_voice("samples/obama.mp3", 5, 6), gap, _voice("samples/bcn_weather.mp3", 0, 6)])
    session = MixedSession(_engine(), classify=classify, echo_gate=EchoGate(SR))
    _run(session, mic, system)

    # Channels finish independently (audio is pushed faster than real time), so check sets
    by_channel = {}
    for s in session.segments:
        by_channel.setdefault(s["channel"], set()).add(s["speaker"])
    assert by_channel["mic"] == {"You"}
    assert by_channel["system"] == ({"Speaker 0", "Speaker 1"} if classify else {"Remote"})
    assert "[You]: " in session.transcript
    assert ("[Speaker 1]: " if classify else "[Remote]: ") in session.transcript
