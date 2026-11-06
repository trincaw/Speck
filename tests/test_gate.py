"""Tests for the pause / resume switch of live transcription and its Space-key binding."""

import os
import pty
import sys
import termios
import threading
import time

import numpy as np
from transformers.audio_utils import load_audio

from speck.asr.base import BaseASREngine, TranscriptionResult
from speck.cli.ui import stop_hint, wait_for_interrupt
from speck.streaming import MixedSession, TranscriptionGate
from speck.streaming.sessions import BaseAudioStreamSession

SR = 16000


def test_gate_toggles_and_reports_changes_only():
    changes = []
    gate = TranscriptionGate(on_change=changes.append)
    assert gate.active

    assert gate.toggle() is False
    assert gate.set(False) is False  # already off: no notification
    assert gate.toggle() is True
    assert gate.set(True) is True
    assert changes == [False, True]


def test_gate_replaces_audio_with_silence_of_the_same_length():
    gate = TranscriptionGate()
    chunk = np.full(160, 0.5, dtype=np.float32)
    assert gate.apply(chunk) is chunk

    gate.set(False)
    out = gate.apply(chunk)
    assert out.shape == chunk.shape and out.dtype == chunk.dtype and not out.any()


def test_gate_survives_a_failing_callback():
    def boom(_):
        raise RuntimeError("boom")

    gate = TranscriptionGate(on_change=boom)
    assert gate.toggle() is False  # the error is logged, not raised


class _Engine:
    sampling_rate = SR


class _Recorder(BaseAudioStreamSession):
    """Session without VAD or ASR that just keeps the audio it was given."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heard = []

    def _run_inference_worker(self):
        cursor = 0
        while (chunk := self._read_available(cursor)) is not None:
            cursor += len(chunk)
            self.heard.append(chunk)


def test_paused_session_hears_silence_and_keeps_its_timeline():
    gate = TranscriptionGate()
    session = _Recorder(_Engine(), gate=gate).start()
    loud = np.full(SR // 10, 0.5, dtype=np.float32)

    session.push_audio(loud)
    gate.set(False)
    session.push_audio(loud)
    gate.set(True)
    session.push_audio(loud)
    session.finish()
    assert session.join(timeout=5)

    heard = np.concatenate(session.heard)
    assert len(heard) == 3 * len(loud)  # nothing dropped: later timestamps stay true
    assert heard[: len(loud)].all() and not heard[len(loud) : 2 * len(loud)].any() and heard[2 * len(loud) :].all()


def test_session_without_gate_is_unchanged():
    session = _Recorder(_Engine()).start()
    loud = np.full(SR // 10, 0.5, dtype=np.float32)
    session.push_audio(loud)
    session.finish()
    assert session.join(timeout=5)
    np.testing.assert_array_equal(np.concatenate(session.heard), loud)


class _StubASR(BaseASREngine):
    name, sampling_rate = "Stub", SR

    def load(self):
        return self

    def transcribe(self, audio, language="auto", skip_special_tokens=True, **kwargs):
        secs = len(audio) / self.sampling_rate
        return TranscriptionResult(f"{secs:.1f}s", "", None, secs, 0.0, 0.0)

    def stream_transcribe(self, audio, **kwargs):
        yield self.transcribe(audio).text


def _speech(start, seconds):
    x = load_audio("samples/obama.mp3", sampling_rate=SR)[int(start * SR) : int((start + seconds) * SR)]
    return (0.5 * x / np.abs(x).max()).astype(np.float32)


def test_speech_while_paused_is_not_transcribed_and_speech_before_it_is():
    """on (a) → off (b) → on (c): a and c are transcribed separately, b never is."""
    a, b, c = _speech(5, 4), _speech(10, 4), _speech(15, 4)
    gate = TranscriptionGate()
    session = _StubASR("stub", device="cpu").create_stream_session(gate=gate).start()

    def push(audio):
        for i in range(0, len(audio), SR // 10):
            session.push_audio(audio[i : i + SR // 10])

    push(a)
    gate.set(False)  # mid-speech: what was already said must still come out
    push(b)
    gate.set(True)
    push(c)
    session.finish()
    assert session.join(timeout=60)

    # Each token is "<seconds>s" of one utterance. Without the pause the three stretches add up
    # to ~9 s of speech (VAD trims pauses); with b muted only a and c remain (~6 s).
    durations = [float(t.strip().rstrip("s")) for t in session._transcript]
    assert len(durations) >= 2
    assert 3 < sum(durations) < 8


def test_mixed_session_hands_the_gate_to_both_channels():
    created = []

    class _FakeEngine:
        def create_stream_session(self, on_token, **kwargs):
            created.append(kwargs)
            return object()

    gate = TranscriptionGate()
    MixedSession(_FakeEngine(), gate=gate)
    assert len(created) == 2 and all(kw["gate"] is gate for kw in created)


def test_space_key_toggles_the_gate_and_the_terminal_is_restored(monkeypatch):
    master, slave = pty.openpty()
    stdin = os.fdopen(slave, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    before = termios.tcgetattr(slave)
    gate = TranscriptionGate()
    deadline = time.monotonic() + 5

    def press():
        time.sleep(0.3)  # after the terminal has been switched to single-key mode
        os.write(master, b"x ")  # other keys are ignored

    threading.Thread(target=press, daemon=True).start()
    try:
        wait_for_interrupt("stopped", gate, until=lambda: not gate.active or time.monotonic() > deadline)
        assert not gate.active
        assert termios.tcgetattr(slave) == before
        assert "Space" in stop_hint(gate)
    finally:
        stdin.close()
        os.close(master)


def test_without_a_terminal_nothing_is_read_and_the_hint_stays_plain(monkeypatch):
    monkeypatch.setattr(sys, "stdin", open(os.devnull))
    gate = TranscriptionGate()
    wait_for_interrupt("stopped", gate, until=lambda: True)
    assert gate.active
    assert stop_hint(gate) == "press Ctrl+C to stop."
