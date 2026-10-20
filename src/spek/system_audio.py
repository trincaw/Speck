"""Linux system-audio capture: transcribe what the desktop is playing (calls, videos, ...).

Works on PulseAudio and PipeWire (via pipewire-pulse) using the stock `pactl` / `parec`
tools, so no extra Python dependencies are needed:

- `ActivityDetector` polls the playback streams (sink inputs) and recognizes a *call*
  when the same application is also recording from a microphone.
- `SystemAudioCapture` records the monitor of the sink that is playing ("what you hear")
  and, during calls, mixes in the microphone ("what you say").
- `DualCapture` records the microphone and system audio as two channels (mixed mode).
- `SystemAudioListener` ties them together: a transcription session starts when audio
  activity starts and is finalized once the desktop has been quiet for `idle_timeout`.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CLIENT_NAME = "spek"
# Streams that are never "real" activity: ourselves, volume meters, notification sounds.
IGNORED_APPS = {CLIENT_NAME, "pavucontrol", "pulseaudio volume control", "pwvucontrol", "plasmapa"}
IGNORED_MEDIA = {"peak detect"}
IGNORED_ROLES = {"event", "notification", "a11y"}


class SystemAudioError(RuntimeError):
    """Raised when the PulseAudio/PipeWire tooling is unavailable or fails."""


def check_available() -> None:
    missing = [tool for tool in ("pactl", "parec") if not shutil.which(tool)]
    if missing:
        raise SystemAudioError(
            f"Missing {', '.join(missing)}: install 'libpulse' (Arch) / 'pulseaudio-utils' (Debian/Ubuntu). "
            "PipeWire needs pipewire-pulse running."
        )


def _pactl_json(*args: str) -> list[dict]:
    try:
        out = subprocess.run(["pactl", "-f", "json", *args], capture_output=True, text=True, check=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as err:
        raise SystemAudioError(f"pactl {' '.join(args)} failed: {err}") from err
    return json.loads(out.stdout or "[]")


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError) as err:
        raise SystemAudioError(f"pactl {' '.join(args)} failed: {err}") from err


# --- stream discovery -----------------------------------------------------------


@dataclass(frozen=True)
class AudioStream:
    """A playback (sink input) or recording (source output) stream of some application."""

    index: int
    device: int  # sink index for playback streams, source index for recording streams
    app: str
    media: str = ""
    binary: str = ""
    app_id: str = ""
    pid: str = ""
    role: str = ""
    corked: bool = False

    @classmethod
    def from_pactl(cls, item: dict, device_key: str) -> AudioStream:
        props = item.get("properties", {})
        return cls(
            index=int(item["index"]),
            device=int(item.get(device_key, -1)),
            app=props.get("application.name", ""),
            media=props.get("media.name", ""),
            binary=props.get("application.process.binary", ""),
            app_id=props.get("pipewire.access.portal.app_id", "") or props.get("application.id", ""),
            pid=props.get("application.process.id", ""),
            role=props.get("media.role", ""),
            corked=bool(item.get("corked", False)),
        )

    @property
    def label(self) -> str:
        name = self.binary or self.app or "unknown"
        return f"{name} ({self.media})" if self.media and self.media != name else name

    @property
    def ignored(self) -> bool:
        return (
            self.app.lower() in IGNORED_APPS
            or self.media.lower() in IGNORED_MEDIA
            or self.role.lower() in IGNORED_ROLES
        )

    def matches(self, patterns: Sequence[str]) -> bool:
        """Case-insensitive substring match against app name, binary, app id and media name."""
        if not patterns:
            return True
        haystack = " ".join((self.app, self.binary, self.app_id, self.media)).lower()
        return any(p.lower() in haystack for p in patterns)

    def same_app(self, other: AudioStream) -> bool:
        if self.pid and self.pid == other.pid:
            return True
        return any(a and a == b for a, b in ((self.binary, other.binary), (self.app_id, other.app_id), (self.app, other.app)))


def playback_streams() -> list[AudioStream]:
    return [AudioStream.from_pactl(i, "sink") for i in _pactl_json("list", "sink-inputs")]


def recording_streams() -> list[AudioStream]:
    return [AudioStream.from_pactl(i, "source") for i in _pactl_json("list", "source-outputs")]


def list_sources() -> list[dict]:
    """Capture devices: microphones and sink monitors (`monitor_of_sink` set)."""
    return _pactl_json("list", "sources")


def sink_monitor(sink_index: Optional[int] = None) -> str:
    """Name of the monitor source for a sink (default sink when None)."""
    if sink_index is None:
        return "@DEFAULT_MONITOR@"
    for sink in _pactl_json("list", "sinks"):
        if int(sink["index"]) == sink_index and sink.get("monitor_source"):
            return sink["monitor_source"]
    return "@DEFAULT_MONITOR@"


def source_name(source_index: Optional[int] = None) -> str:
    """Name of a capture source (default microphone when None)."""
    if source_index is not None:
        for source in list_sources():
            if int(source["index"]) == source_index:
                return source["name"]
    return "@DEFAULT_SOURCE@"


# --- activity detection ---------------------------------------------------------


@dataclass
class Activity:
    """Snapshot of what the desktop is doing audio-wise."""

    streams: list[AudioStream] = field(default_factory=list)  # active, matching playback streams
    call_streams: list[AudioStream] = field(default_factory=list)  # their mic recordings, if in a call

    @property
    def active(self) -> bool:
        return bool(self.streams)

    @property
    def in_call(self) -> bool:
        return bool(self.call_streams)

    @property
    def sink(self) -> Optional[int]:
        return self.streams[0].device if self.streams else None

    @property
    def mic_source(self) -> Optional[int]:
        return self.call_streams[0].device if self.call_streams else None

    @property
    def label(self) -> str:
        names = list(dict.fromkeys(s.binary or s.app for s in self.streams))
        return ", ".join(names) or "system audio"


class ActivityDetector:
    """Decide from the current streams whether something worth transcribing is playing."""

    def __init__(self, apps: Sequence[str] = ()):
        self.apps = list(apps)

    def evaluate(self, playback: Sequence[AudioStream], recording: Sequence[AudioStream]) -> Activity:
        streams = [s for s in playback if not s.corked and not s.ignored and s.matches(self.apps)]
        mics = [r for r in recording if not r.corked and not r.ignored]
        calls = [r for r in mics if any(r.same_app(s) for s in streams)]
        return Activity(streams=streams, call_streams=calls)

    def poll(self) -> Activity:
        return self.evaluate(playback_streams(), recording_streams())


# --- capture --------------------------------------------------------------------


class ParecReader:
    """Raw mono float32 PCM from a PulseAudio/PipeWire source via `parec`."""

    def __init__(self, device: str, sample_rate: int = 16000, block_ms: int = 100):
        self.device = device
        self.sample_rate = sample_rate
        self.block_bytes = sample_rate * block_ms // 1000 * 4
        cmd = ["parec", "--raw", "--format=float32le", f"--rate={sample_rate}", "--channels=1",
               f"--latency-msec={block_ms}", f"--client-name={CLIENT_NAME}", "--stream-name=capture",
               f"--device={device}"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    def read(self) -> Optional[np.ndarray]:
        """Block for the next chunk; None once the stream ends."""
        data = self._proc.stdout.read(self.block_bytes) if self._proc.stdout else b""
        usable = len(data) - len(data) % 4
        return np.frombuffer(data[:usable], dtype="<f4").copy() if usable else None

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()


class SystemAudioCapture:
    """Records a source (normally a sink monitor) and optionally mixes in a microphone.

    The monitor drives the clock; mic audio is buffered and added sample-aligned, with
    the backlog capped so clock drift between the two devices can't build latency.
    """

    MAX_MIC_BACKLOG_S = 0.5

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        device: str,
        sample_rate: int = 16000,
        block_ms: int = 100,
        reader_factory: Callable[..., ParecReader] = ParecReader,
    ):
        self.on_audio = on_audio
        self.device = device
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._reader_factory = reader_factory
        self._main: Optional[ParecReader] = None
        self._mic: Optional[ParecReader] = None
        self._mic_device: Optional[str] = None
        self._mic_buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    @property
    def mic_device(self) -> Optional[str]:
        return self._mic_device

    def start(self) -> SystemAudioCapture:
        self._main = self._reader_factory(self.device, sample_rate=self.sample_rate, block_ms=self.block_ms)
        self._spawn(self._run_main)
        return self

    def set_mic(self, device: Optional[str]) -> None:
        """Start mixing `device` into the capture, or stop mixing when None."""
        if device == self._mic_device:
            return
        old, self._mic = self._mic, None
        if old:
            old.close()
        with self._lock:
            self._mic_buf = np.zeros(0, dtype=np.float32)
        self._mic_device = device
        if device:
            self._mic = self._reader_factory(device, sample_rate=self.sample_rate, block_ms=self.block_ms)
            self._spawn(self._run_mic, self._mic)

    def stop(self) -> None:
        self.set_mic(None)
        if self._main:
            self._main.close()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()

    def _spawn(self, target, *args) -> None:
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()
        self._threads.append(t)

    def _run_mic(self, reader: ParecReader) -> None:
        cap = int(self.MAX_MIC_BACKLOG_S * self.sample_rate)
        while self._mic is reader and (chunk := reader.read()) is not None:
            with self._lock:
                self._mic_buf = np.concatenate([self._mic_buf, chunk])[-cap:]

    def _take_mic(self, n: int) -> Optional[np.ndarray]:
        if not self._mic:
            return None
        with self._lock:
            taken, self._mic_buf = self._mic_buf[:n], self._mic_buf[n:]
        return np.pad(taken, (0, n - len(taken)))

    def mix(self, block: np.ndarray) -> np.ndarray:
        mic = self._take_mic(len(block))
        return block if mic is None else np.clip(block + mic, -1.0, 1.0)

    def _run_main(self) -> None:
        while self._main and (block := self._main.read()) is not None:
            try:
                self.on_audio(self.mix(block))
            except Exception as err:
                logger.error("System audio consumer error: %s", err)


class DualCapture:
    """Records the microphone and system audio as two separate channels (mixed mode)."""

    def __init__(
        self,
        on_mic: Callable[[np.ndarray], None],
        on_system: Callable[[np.ndarray], None],
        mic_device: str = "@DEFAULT_SOURCE@",
        system_device: str = "@DEFAULT_MONITOR@",
        sample_rate: int = 16000,
        block_ms: int = 100,
        reader_factory: Callable[..., ParecReader] = ParecReader,
    ):
        self.channels = {"mic": (mic_device, on_mic), "system": (system_device, on_system)}
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self._reader_factory = reader_factory
        self._readers: list[ParecReader] = []
        self._threads: list[threading.Thread] = []

    def start(self) -> DualCapture:
        for name, (device, consumer) in self.channels.items():
            reader = self._reader_factory(device, sample_rate=self.sample_rate, block_ms=self.block_ms)
            thread = threading.Thread(target=self._pump, args=(name, reader, consumer), daemon=True)
            self._readers.append(reader)
            self._threads.append(thread)
            thread.start()
        return self

    def stop(self) -> None:
        for reader in self._readers:
            reader.close()
        for thread in self._threads:
            thread.join(timeout=2)
        self._readers.clear()
        self._threads.clear()

    @staticmethod
    def _pump(name: str, reader: ParecReader, consumer: Callable[[np.ndarray], None]) -> None:
        while (block := reader.read()) is not None:
            try:
                consumer(block)
            except Exception as err:
                logger.error("%s audio consumer error: %s", name, err)


# --- orchestration --------------------------------------------------------------


@dataclass
class Recording:
    """One transcription session, from activity start to idle timeout."""

    session: object  # BaseAudioStreamSession
    capture: SystemAudioCapture
    label: str
    started_at: float  # wall clock (time.time)
    in_call: bool = False


class SystemAudioListener:
    """State machine: idle -> recording when audio activity starts, back to idle when quiet.

    `mic`: "auto" mixes the microphone in only while a call is detected, "on" always,
    "off" never. Drive it with `run()`, or call `step()` yourself (e.g. from tests).
    """

    def __init__(
        self,
        session_factory: Callable[[], object],
        detector: Optional[ActivityDetector] = None,
        sample_rate: int = 16000,
        mic: str = "auto",
        idle_timeout: float = 8.0,
        min_active: float = 1.0,
        poll_interval: float = 0.5,
        device: Optional[str] = None,
        on_start: Optional[Callable[[Recording], None]] = None,
        on_end: Optional[Callable[[Recording], None]] = None,
        capture_factory: Callable[..., SystemAudioCapture] = SystemAudioCapture,
        resolve_monitor: Callable[[Optional[int]], str] = sink_monitor,
        resolve_mic: Callable[[Optional[int]], str] = source_name,
    ):
        if mic not in ("auto", "on", "off"):
            raise ValueError("mic must be 'auto', 'on' or 'off'")
        self.session_factory = session_factory
        self.detector = detector or ActivityDetector()
        self.sample_rate = sample_rate
        self.mic = mic
        self.idle_timeout = idle_timeout
        self.min_active = min_active
        self.poll_interval = poll_interval
        self.device = device  # fixed capture device: record continuously, no detection
        self.on_start = on_start
        self.on_end = on_end
        self._capture_factory = capture_factory
        self._resolve_monitor = resolve_monitor
        self._resolve_mic = resolve_mic
        self.recording: Optional[Recording] = None
        self._active_since: Optional[float] = None
        self._quiet_since: Optional[float] = None

    def step(self, activity: Activity, now: float) -> None:
        if self.recording is None:
            if not activity.active:
                self._active_since = None
            elif self._active_since is None:
                self._active_since = now
            if self._active_since is not None and now - self._active_since >= self.min_active:
                self._begin(activity)
            return

        if activity.active:
            self._quiet_since = None
        elif self._quiet_since is None:
            self._quiet_since = now
        if self._quiet_since is not None and now - self._quiet_since >= self.idle_timeout:
            self._end()
            return
        if activity.active:
            self._update_mic(activity)

    def run(self, stop: threading.Event) -> None:
        if self.device:
            self._begin(Activity(), label=self.device)
            stop.wait()
        else:
            while not stop.is_set():
                try:
                    self.step(self.detector.poll(), time.monotonic())
                except SystemAudioError as err:
                    logger.warning("%s", err)
                stop.wait(self.poll_interval)
        self.close()

    def close(self) -> None:
        if self.recording:
            self._end()

    def _mic_for(self, activity: Activity) -> Optional[str]:
        if self.mic == "on" or (self.mic == "auto" and activity.in_call):
            return self._resolve_mic(activity.mic_source)
        return None

    def _update_mic(self, activity: Activity) -> None:
        rec = self.recording
        rec.capture.set_mic(self._mic_for(activity))
        rec.in_call = rec.in_call or activity.in_call

    def _begin(self, activity: Activity, label: Optional[str] = None) -> None:
        session = self.session_factory()
        device = self.device or self._resolve_monitor(activity.sink)
        capture = self._capture_factory(session.push_audio, device, sample_rate=self.sample_rate)
        self.recording = Recording(session, capture, label or activity.label, time.time(), activity.in_call)
        session.start()
        capture.start()
        capture.set_mic(self._mic_for(activity))
        self._quiet_since = None
        if self.on_start:
            self.on_start(self.recording)

    def _end(self) -> None:
        rec, self.recording = self.recording, None
        self._active_since = self._quiet_since = None
        rec.capture.stop()
        rec.session.finish()
        rec.session.join(timeout=60)
        if self.on_end:
            self.on_end(rec)
