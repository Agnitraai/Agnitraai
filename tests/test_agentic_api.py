import importlib
import json
import os
import time

import pytest
from starlette.testclient import TestClient

from agnitra.api import app as app_module
from agnitra.api.app import create_app


def _sample_graph():
    return [
        {
            "name": "matmul_main",
            "op": "matmul",
            "shape": [64, 64],
            "cuda_time_ms": 12.5,
        },
        {
            "name": "relu_out",
            "op": "relu",
            "cuda_time_ms": 2.1,
        },
    ]


def _sample_telemetry():
    return {
        "events": [
            {"name": "aten::matmul", "cuda_time_total": 12.5},
            {"name": "aten::relu", "cuda_time_total": 2.1},
        ]
    }


def test_optimize_endpoint_returns_expected_payload():
    app = create_app()
    client = TestClient(app)

    files = {
        "model_graph": ("graph.json", json.dumps(_sample_graph()), "application/json"),
        "telemetry": ("telemetry.json", json.dumps(_sample_telemetry()), "application/json"),
    }

    response = client.post("/optimize", data={"target": "A100"}, files=files)
    assert response.status_code == 200
    payload = response.json()

    assert payload["target"] == "A100"
    assert "ir_graph" in payload and "nodes" in payload["ir_graph"]
    optimized_nodes = payload["ir_graph"]["nodes"]
    assert any(node.get("annotations", {}).get("status") == "optimized" for node in optimized_nodes)

    kernel = payload.get("kernel", {})
    assert kernel.get("source")
    assert "run_kernel" in kernel["source"]

    instructions = payload.get("patch_instructions", [])
    assert instructions
    assert instructions[0]["order"] == 1
    assert payload["bottleneck"]["name"] == "matmul_main"


def test_optimize_endpoint_requires_target_field():
    app = create_app()
    client = TestClient(app)

    files = {
        "model_graph": ("graph.json", json.dumps(_sample_graph()), "application/json"),
        "telemetry": ("telemetry.json", json.dumps(_sample_telemetry()), "application/json"),
    }

    response = client.post("/optimize", files=files)
    assert response.status_code == 400


def test_optimize_accepts_json_body():
    app = create_app()
    client = TestClient(app)

    payload = {
        "target": "H100",
        "model_graph": _sample_graph(),
        "telemetry": _sample_telemetry(),
    }

    response = client.post("/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target"] == "H100"
    assert data["bottleneck"]["expected_speedup_pct"] >= 5.0


def _reload_app_module():
    importlib.reload(app_module)
    return app_module


@pytest.fixture
def reload_app(monkeypatch):
    original_env = dict(os.environ)
    try:
        yield _reload_app_module
    finally:
        os.environ.clear()
        os.environ.update(original_env)
        importlib.reload(app_module)


def test_optimize_requires_api_key_when_configured(monkeypatch, reload_app):
    monkeypatch.setenv("AGNITRA_API_KEY", "super-secret")
    module = reload_app()
    app = module.create_app()
    client = TestClient(app)
    payload = {
        "target": "A100",
        "model_graph": _sample_graph(),
        "telemetry": _sample_telemetry(),
    }

    response = client.post("/optimize", json=payload)
    assert response.status_code == 401

    response = client.post("/optimize", json=payload, headers={"x-api-key": "super-secret"})
    assert response.status_code == 200


def test_async_optimize_queues_job(monkeypatch):
    monkeypatch.delenv("AGNITRA_API_KEY", raising=False)
    importlib.reload(app_module)
    def _fake_run(model_graph, telemetry, target, **kwargs):
        return {
            "target": target,
            "telemetry_summary": {"event_count": 1},
            "bottleneck": {"baseline_latency_ms": 10.0, "expected_latency_ms": 5.0, "expected_speedup_pct": 50.0},
            "ir_graph": {"nodes": [], "metadata": {"target": target, "node_count": 0}},
            "kernel": {"source": "kernel"},
            "patch_instructions": [],
        }

    monkeypatch.setattr(app_module, "run_agentic_optimization", _fake_run)
    app = app_module.create_app()
    client = TestClient(app)

    payload = {
        "target": "A100",
        "model_graph": _sample_graph(),
        "telemetry": _sample_telemetry(),
        "async": True,
        "project_id": "proj-async",
        "model_name": "demo",
    }

    response = client.post("/optimize", json=payload)
    assert response.status_code == 202
    job_info = response.json()
    job_id = job_info.get("job_id")
    assert job_id

    final = None
    for _ in range(40):
        result = client.get(f"/jobs/{job_id}")
        assert result.status_code == 200
        body = result.json()
        status = body.get("status")
        if status == "completed":
            final = body
            break
        if status == "failed":
            pytest.fail(f"Queued job failed: {body.get('error')}")
        time.sleep(0.05)

    assert final is not None, "Expected job to complete within allotted time"
    assert final["status"] == "completed"
    assert "result" in final
    assert final["result"]["target"] == "A100"


def test_job_status_requires_valid_identifier():
    importlib.reload(app_module)
    app = app_module.create_app()
    client = TestClient(app)

    response = client.get("/jobs/unknown-job")
    assert response.status_code == 404
