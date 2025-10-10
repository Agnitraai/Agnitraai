"""Tests for the Agnitra dashboard FastAPI application."""

from __future__ import annotations

import io
import json
import zipfile

from starlette.testclient import TestClient

from agnitra.dashboard import create_app


def test_dashboard_upload_and_summary_roundtrip() -> None:
    app = create_app()
    client = TestClient(app)

    # Landing page renders without uploaded state.
    response = client.get("/")
    assert response.status_code == 200

    baseline = {
        "latency_ms": 83.2,
        "tokens_per_sec": 3149.0,
        "gpu": {"gpu_utilisation": 0.71},
        "events": [
            {"name": "decoder.layer1", "latency_ms": 20.0, "shape": [1, 4096]},
            {"name": "decoder.layer2", "latency_ms": 18.5, "shape": [1, 4096]},
        ],
    }
    optimized = {
        "latency_ms": 62.9,
        "tokens_per_sec": 4168.0,
        "gpu": {"gpu_utilisation": 0.66},
        "events": [
            {"name": "decoder.layer1", "latency_ms": 14.2, "shape": [1, 4096]},
            {"name": "decoder.layer2", "latency_ms": 15.1, "shape": [1, 4096]},
        ],
    }
    usage_event = {
        "project_id": "demo",
        "model_name": "TinyLlama",
        "baseline_latency_ms": 83.2,
        "optimized_latency_ms": 62.9,
        "baseline_tokens_per_sec": 3149.0,
        "optimized_tokens_per_sec": 4168.0,
        "gpu_hours_before": 0.0231,
        "gpu_hours_after": 0.0175,
        "cost_before": 0.0578,
        "cost_after": 0.0437,
    }

    files = {
        "telemetry_before": (
            "baseline.json",
            json.dumps(baseline),
            "application/json",
        ),
        "telemetry_after": (
            "optimized.json",
            json.dumps(optimized),
            "application/json",
        ),
        "usage_event_file": (
            "usage.json",
            json.dumps(usage_event),
            "application/json",
        ),
        "optimized_ir": ("optimized.mlir", "module {}", "text/plain"),
        "kernel_file": ("kernel.triton", "fn kernel() {}", "text/plain"),
    }

    response = client.post(
        "/upload",
        data={"model_name": "TinyLlama", "notes": "Profiling run"},
        files=files,
        follow_redirects=False,
    )
    # Redirect after upload to retain the active tab.
    assert response.status_code == 303

    summary = client.get("/api/summary").json()
    metrics = {metric["label"]: metric for metric in summary["metrics"]}
    assert metrics["Latency (ms)"]["baseline"] == 83.2
    assert metrics["Latency (ms)"]["optimized"] == 62.9
    assert metrics["Tokens / second"]["optimized"] == 4168.0

    layers = client.get("/api/model-analyzer").json()["layers"]
    first_layer = {layer["name"]: layer for layer in layers}["decoder.layer1"]
    assert first_layer["baseline_latency_ms"] == 20.0
    assert first_layer["optimized_latency_ms"] == 14.2

    artifacts = client.get("/api/kernel-artifacts").json()["artifacts"]
    assert len(artifacts) == 2
    download_response = client.get(artifacts[0]["download_url"])
    assert download_response.status_code == 200
    assert download_response.content

    analyzer_html = client.get("/?tab=model-analyzer").text
    assert "decoder.layer1" in analyzer_html
    assert "&#34;" not in analyzer_html

    export_response = client.get("/export/sdk-pack")
    assert export_response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(export_response.content), mode="r") as archive:
        names = set(archive.namelist())
        assert "summary.json" in names
        assert "telemetry/baseline.json" in names
        assert "telemetry/optimized.json" in names
        summary_payload = json.loads(archive.read("summary.json"))
        assert summary_payload["metrics"]
