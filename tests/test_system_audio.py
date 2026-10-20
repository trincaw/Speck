"""Tests for Linux system-audio detection, mixing and the listener state machine (no audio server needed)."""

import numpy as np
import pytest

from speck.system_audio import Activity, ActivityDetector, AudioStream, SystemAudioCapture, SystemAudioListener


def _pactl_item(index, device, app, binary="", pid="1", corked=False, media="playStream", role=None):
    props = {"application.name": app, "application.process.binary": binary,
             "application.process.id": pid, "media.name": media}
    if role:
        props["media.role"] = role
    return {"index": index, "sink": device, "source": device, "corked": corked, "properties": props}


def _play(*args, **kw):
    return AudioStream.from_pactl(_pactl_item(*args, **kw), "sink")


def _rec(*args, **kw):
    return AudioStream.from_pactl(_pactl_item(*args, **kw), "source")


def test_detects_call_when_app_plays_and_records():
    discord = _play(421, 60, "WEBRTC VoiceEngine", "Discord", pid="108")
    discord_mic = _rec(426, 56, "WEBRTC VoiceEngine", "Discord", pid="108", media="recStream")
    activity = ActivityDetector().evaluate([discord], [discord_mic])
    assert activity.active and activity.in_call
    assert activity.sink == 60 and activity.mic_source == 56
    assert activity.label == "Discord"


def test_video_is_active_but_not_a_call():
    firefox = _play(1, 60, "Firefox", "firefox", pid="10")
    other_mic = _rec(2, 56, "OBS", "obs", pid="20")
    activity = ActivityDetector().evaluate([firefox], [other_mic])
    assert activity.active and not activity.in_call


def test_ignores_paused_notifications_own_and_unmatched_streams():
    streams = [
        _play(1, 60, "Firefox", "firefox", corked=True),
        _play(2, 60, "Slack", "slack", role="event"),
        _play(3, 60, "speck", "parec"),
        _play(4, 60, "Spotify", "spotify"),
    ]
    assert not ActivityDetector().evaluate(streams[:3], []).active
    assert not ActivityDetector(apps=["discord"]).evaluate(streams, []).active
    assert ActivityDetector(apps=["SPOT"]).evaluate(streams, []).label == "spotify"


class _FakeReader:
    def __init__(self, device, sample_rate=16000, block_ms=100):
        self.device = device

    def read(self):
        return None

    def close(self):
        pass


def test_capture_mixes_mic_aligned_and_padded():
    cap = SystemAudioCapture(lambda _: None, "mon", reader_factory=_FakeReader)
    block = np.full(4, 0.5, dtype=np.float32)
    np.testing.assert_array_equal(cap.mix(block), block)  # no mic: passthrough

    cap.set_mic("mic")
    cap._mic_buf = np.array([0.25, 0.75, 0.1], dtype=np.float32)
    np.testing.assert_allclose(cap.mix(block), [0.75, 1.0, 0.6, 0.5])  # clipped, zero-padded
    cap.stop()
    assert cap.mic_device is None


class _FakeSession:
    def __init__(self):
        self.started = self.finished = False
        self.transcript = ""

    def start(self):
        self.started = True
        return self

    def push_audio(self, chunk):
        pass

    def finish(self):
        self.finished = True

    def join(self, timeout=None):
        return True


class _FakeCapture:
    def __init__(self, on_audio, device, sample_rate=16000):
        self.device, self.mic_device, self.stopped = device, None, False

    def start(self):
        return self

    def set_mic(self, device):
        self.mic_device = device

    def stop(self):
        self.stopped = True


@pytest.fixture
def listener():
    events = []
    lst = SystemAudioListener(
        _FakeSession,
        idle_timeout=5,
        min_active=1,
        capture_factory=_FakeCapture,
        resolve_monitor=lambda sink: f"monitor-{sink}",
        resolve_mic=lambda src: f"mic-{src}",
        on_start=lambda r: events.append(("start", r)),
        on_end=lambda r: events.append(("end", r)),
    )
    lst.events = events
    return lst


def test_listener_starts_after_min_active_and_stops_after_idle(listener):
    video = Activity(streams=[_play(1, 60, "Firefox", "firefox")])
    listener.step(video, 0.0)
    listener.step(video, 0.5)
    assert listener.recording is None  # debounced

    listener.step(video, 1.0)
    rec = listener.recording
    assert rec and rec.session.started and rec.capture.device == "monitor-60"
    assert rec.capture.mic_device is None  # not a call

    listener.step(Activity(), 2.0)
    listener.step(video, 4.0)  # resumed before the timeout: same recording
    listener.step(Activity(), 5.0)
    listener.step(Activity(), 9.9)
    assert listener.recording is rec

    listener.step(Activity(), 10.0)
    assert listener.recording is None
    assert rec.session.finished and rec.capture.stopped
    assert [e for e, _ in listener.events] == ["start", "end"]


def test_listener_mixes_mic_only_during_calls(listener):
    play = _play(1, 60, "WEBRTC VoiceEngine", "Discord", pid="7")
    call = Activity(streams=[play], call_streams=[_rec(2, 56, "WEBRTC VoiceEngine", "Discord", pid="7")])
    listener.step(Activity(streams=[play]), 0.0)
    listener.step(Activity(streams=[play]), 1.0)
    assert listener.recording.capture.mic_device is None

    listener.step(call, 2.0)
    assert listener.recording.capture.mic_device == "mic-56"
    assert listener.recording.in_call

    listener.mic = "off"
    listener.step(call, 3.0)
    assert listener.recording.capture.mic_device is None
