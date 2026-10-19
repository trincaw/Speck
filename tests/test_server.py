"""Tests for FastAPI server endpoints."""

from fastapi.testclient import TestClient
from spek.server import app

client = TestClient(app)


def test_ui_page():
    response = client.get("/")
    assert response.status_code == 200
    assert "Nemotron 3.5 ASR Streaming" in response.text
    assert "Start Live Mic" in response.text


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "nvidia/nemotron-3.5-asr-streaming-0.6b" in data["model_id"]
    assert "cuda" in data["device"] or "cpu" in data["device"]
