"""FastAPI & WebSocket server for live ASR streaming with a browser UI."""

from __future__ import annotations

import asyncio
import html
import io
import json
import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse

from spek import __version__
from spek.base import BaseASREngine
from spek.config import load_env_file
from spek.factory import get_engine as create_engine, list_available_models
from spek.streamer import BaseAudioStreamSession

load_env_file()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Spek ASR Streaming Server",
    description="Model-Agnostic Real-Time Speech Recognition API & Web UI",
    version=__version__,
)

_engine: Optional[BaseASREngine] = None
_engine_lock = threading.Lock()
_UI_TEMPLATE = (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


def get_engine() -> BaseASREngine:
    """Return the process-wide engine, creating and loading it once."""
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = create_engine().load()
        return _engine


def set_engine(engine: BaseASREngine) -> None:
    global _engine
    with _engine_lock:
        _engine = engine


@lru_cache(maxsize=1)
def _speaker_embedder(sample_rate: int):
    from spek.speakers import SpeakerEmbedder
    return SpeakerEmbedder(sample_rate=sample_rate)  # thread-safe, shared by all sessions


def _speaker_classifier(config: dict[str, Any], sample_rate: int):
    """Fresh per-connection classifier (speaker identities are not shared between streams)."""
    from spek.config import get_speaker_threshold
    from spek.speakers import SpeakerClassifier
    return SpeakerClassifier(
        _speaker_embedder(sample_rate),
        threshold=float(config.get("speaker_threshold", get_speaker_threshold())),
        max_speakers=int(config.get("num_speakers", 0)),
    )


@lru_cache(maxsize=8)
def _render_ui(engine: BaseASREngine) -> str:
    def option(value: Any, label: str, selected: bool = False) -> str:
        return f'<option value="{html.escape(str(value))}"{" selected" if selected else ""}>{html.escape(label)}</option>'

    languages = [option("auto", "Auto-Detect Language")]
    for tier in engine.supported_languages.values():
        languages.append(f'<optgroup label="{html.escape(tier.name)}">')
        languages += [option(loc, loc, loc in ("en-US", "en")) for loc in tier.locales]
        languages.append("</optgroup>")

    presets = [
        option(k, f"{k} tokens - {v.get('description', '')}", k == 6)
        for k, v in engine.streaming_presets.items()
    ] or [option(6, "Default Latency Preset", True)]

    replacements = {
        "{{ENGINE_NAME}}": html.escape(engine.name),
        "{{MODEL_ID}}": html.escape(engine.model_id),
        "{{LANGUAGE_OPTIONS}}": "\n".join(languages),
        "{{PRESET_OPTIONS}}": "\n".join(presets),
    }
    page = _UI_TEMPLATE
    for key, value in replacements.items():
        page = page.replace(key, value)
    return page


def _decode_upload(contents: bytes, target_sr: int) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(contents), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != target_sr:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(_render_ui(get_engine()))


@app.get("/health")
async def health():
    engine = get_engine()
    cuda = torch.cuda.is_available()
    return {
        "status": "healthy",
        "engine_name": engine.name,
        "model_id": engine.model_id,
        "backend": getattr(engine, "backend", "local"),
        "device": engine.device,
        "dtype": str(engine.dtype),
        "cuda_available": cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
    }


@app.get("/models")
async def get_models():
    engine = get_engine()
    return {"models": list_available_models(), "active_model": engine.model_id, "active_engine": engine.name}


@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    skip_special_tokens: bool = Form(True),
):
    """Batch transcription of an uploaded audio file (runs off the event loop)."""
    engine = get_engine()
    wav = await run_in_threadpool(_decode_upload, await file.read(), engine.sampling_rate)
    result = await run_in_threadpool(
        engine.transcribe, wav, language=language, skip_special_tokens=skip_special_tokens
    )
    return {
        "text": result.text,
        "raw_text": result.raw_text,
        "detected_language": result.detected_language,
        "duration_seconds": result.duration_seconds,
        "inference_seconds": result.inference_seconds,
        "real_time_factor": result.real_time_factor,
        "segments": result.segments,
    }


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """Live streaming: first a JSON config message, then float32 16 kHz PCM frames, then {"type": "eof"}.

    Server messages: {"type": "token"} and finally {"type": "done"}. With `"speakers": true`
    in the config, text is speaker-tagged and the server also sends {"type": "segment"} per
    finished turn and {"type": "speaker"} (provisional label while someone is talking);
    "done" then carries the corrected `segments`.
    """
    await websocket.accept()
    engine = get_engine()
    loop = asyncio.get_running_loop()
    session: Optional[BaseAudioStreamSession] = None

    def send(payload: dict[str, Any]) -> None:
        asyncio.run_coroutine_threadsafe(websocket.send_json(payload), loop)

    try:
        config = json.loads(await websocket.receive_text())
        speaker_opts: dict[str, Any] = {}
        if config.get("speakers"):
            speaker_opts = {
                "speakers": await run_in_threadpool(_speaker_classifier, config, engine.sampling_rate),
                "on_segment": lambda seg: send({"type": "segment", **seg}),
                "on_speaker": lambda m: send({"type": "speaker", "label": m.label, "score": round(m.score, 3)}),
            }
        session = await run_in_threadpool(
            engine.create_stream_session,
            language=config.get("language", "auto"),
            lookahead=int(config.get("lookahead", 6)),
            skip_special_tokens=bool(config.get("skip_special_tokens", True)),
            on_token=lambda token: send({"type": "token", "text": token}),
            **speaker_opts,
        )
        session.start()

        while True:
            message = await websocket.receive()
            if message.get("bytes"):
                session.push_audio(np.frombuffer(message["bytes"], dtype=np.float32))
            elif message.get("text") and json.loads(message["text"]).get("type") == "eof":
                break
            elif message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect()

        session.finish()
        await asyncio.to_thread(session.join, 30.0)  # wait for trailing tokens
        done = {"type": "done", "transcript": session.transcript}
        if speaker_opts:
            done["segments"] = session.segments
        await websocket.send_json(done)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected.")
    except Exception as err:
        logger.error("WebSocket streaming error: %s", err, exc_info=True)
    finally:
        if session:
            session.stop()
