"""Integration tests for FastAPI endpoints, streaming generation, and Typer CLI."""

import json

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from prompt_manifold.api.app import app
from prompt_manifold.cli.main import app as cli_app
from prompt_manifold.streaming.streamer import PromptStreamer


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def runner():
    return CliRunner()


def test_api_health_check(client):
    """GET /v1/health should return service health and version."""
    res = client.get("/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "prompt-manifold"


def test_api_enhance_endpoint(client):
    """POST /v1/enhance should project sparse prompt to in-distribution manifold."""
    payload = {
        "prompt": "a teddy bear reading a book",
        "target": "flux",
        "detail_level": 3,
        "generate_negative": True,
    }
    res = client.post("/v1/enhance", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "bear" in data["enhanced_prompt"].lower()
    assert data["in_distribution_score"] > 0.6
    assert data["expansion_ratio"] > 1.5
    assert data["negative_prompt"] is not None


def test_api_enhance_empty_prompt_400(client):
    """POST /v1/enhance should reject empty prompt with HTTP 400."""
    res = client.post("/v1/enhance", json={"prompt": "   ", "target": "flux"})
    assert res.status_code == 400


def test_api_benchmark_endpoint(client):
    """POST /v1/benchmark should return comparative empirical report."""
    res = client.post("/v1/benchmark")
    assert res.status_code == 200
    data = res.json()
    assert data["scenario_count"] > 0
    assert data["win_rate_percentage"] == 100.0


@pytest.mark.anyio
async def test_prompt_streamer_tokens():
    """PromptStreamer should asynchronously stream tokens."""
    streamer = PromptStreamer()
    tokens = []
    async for tok in streamer.stream_tokens("a red car", chunk_delay_ms=1.0):
        tokens.append(tok)

    assert len(tokens) > 3
    reconstructed = "".join(tokens)
    assert "car" in reconstructed.lower()


@pytest.mark.anyio
async def test_prompt_streamer_events():
    """PromptStreamer should yield SSE events with start, token, and complete."""
    streamer = PromptStreamer()
    events = []
    async for ev in streamer.stream_events("cyberpunk samurai", chunk_delay_ms=1.0):
        events.append(ev)

    event_names = [e["event"] for e in events]
    assert "start" in event_names
    assert "token" in event_names
    assert "complete" in event_names

    complete_data = json.loads([e for e in events if e["event"] == "complete"][0]["data"])
    assert "enhanced_prompt" in complete_data
    assert complete_data["in_distribution_score"] > 0.5


def test_cli_enhance_command(runner):
    """CLI 'enhance' command should execute without error."""
    result = runner.invoke(cli_app, ["enhance", "a teddy bear reading a book", "--target", "flux"])
    assert result.exit_code == 0
    assert "PromptManifold Projection" in result.stdout


def test_cli_inspect_command(runner):
    """CLI 'inspect' command should decompose intent slots."""
    result = runner.invoke(cli_app, ["inspect", "portrait of an astronaut in space"])
    assert result.exit_code == 0
    assert "Semantic Slot Decomposition" in result.stdout


def test_cli_benchmark_command(runner):
    """CLI 'benchmark' command should display empirical report."""
    result = runner.invoke(cli_app, ["benchmark"])
    assert result.exit_code == 0
    assert "Benchmark Report" in result.stdout


def test_cli_version_command(runner):
    """CLI 'version' command should display version."""
    result = runner.invoke(cli_app, ["version"])
    assert result.exit_code == 0
    assert "PromptManifold v" in result.stdout
