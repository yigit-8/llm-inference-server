"""
The API surface, without a model.

Nothing asserted here depends on the weights: request validation, health
reporting and the shape of the responses are the server's own behaviour. The
runner is therefore stubbed and the lifespan is never run, so these tests do not
download or load anything.
"""

import pytest
from fastapi.testclient import TestClient

from src import serve
from src.config import settings


class StubRunner:
    """Stands in for EngineRunner: only what the handlers actually call."""

    def __init__(self, healthy: bool = True, error_message: str | None = None) -> None:
        self._healthy = healthy
        self.error_message = error_message

    def is_healthy(self) -> bool:
        return self._healthy

    def stats(self) -> dict:
        return {
            "model": settings.MODEL_NAME,
            "max_batch_size": settings.MAX_BATCH_SIZE,
            "running": 2,
            "queued": 1,
        }


def make_client(monkeypatch, runner: StubRunner) -> TestClient:
    # Deliberately not used as a context manager: entering it would run the
    # lifespan, which loads a real model.
    monkeypatch.setattr(serve, "runner", runner)
    return TestClient(serve.app)


@pytest.fixture()
def client(monkeypatch):
    return make_client(monkeypatch, StubRunner())


def test_health_reports_the_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": settings.MODEL_NAME}


def test_stats_reports_the_engine_state(client):
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json() == {
        "model": settings.MODEL_NAME,
        "max_batch_size": settings.MAX_BATCH_SIZE,
        "running": 2,
        "queued": 1,
    }


def test_empty_prompt_is_rejected(client):
    assert client.post("/generate", json={"prompt": ""}).status_code == 422


def test_oversized_prompt_is_rejected_before_tokenization(client):
    """The body is bounded by pydantic, so a megabyte of text never reaches the
    tokenizer or the engine queue."""
    response = client.post("/generate", json={"prompt": "x" * 8001})
    assert response.status_code == 422


def test_health_is_503_when_the_engine_thread_has_died(monkeypatch):
    runner = StubRunner(healthy=False, error_message="RuntimeError: cache exploded")
    response = make_client(monkeypatch, runner).get("/health")

    assert response.status_code == 503
    assert "cache exploded" in response.json()["detail"]
