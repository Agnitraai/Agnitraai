from agnitra.api.app import create_app
from agnitra.core.metering import UsageMeter
from agnitra.core.runtime import OptimizationSnapshot
from starlette.testclient import TestClient


def _snapshot(latency: float, tps: float, tokens: int, gpu_util: float = 50.0) -> OptimizationSnapshot:
    return OptimizationSnapshot(
        latency_ms=latency,
        tokens_per_sec=tps,
        tokens_processed=tokens,
        gpu_utilization=gpu_util,
    )


def test_usage_endpoint_generates_event_from_snapshots():
    app = create_app()
    client = TestClient(app)

    payload = {
        "project_id": "proj-123",
        "model_name": "demo-model",
        "baseline": {
            "latency_ms": 120.0,
            "tokens_per_sec": 90.0,
            "tokens_processed": 2048,
            "gpu_utilization": 70.0,
        },
        "optimized": {
            "latency_ms": 80.0,
            "tokens_per_sec": 135.0,
            "tokens_processed": 2048,
            "gpu_utilization": 60.0,
        },
        "tokens_processed": 2048,
        "metadata": {"plan": "enterprise"},
    }

    response = client.post("/usage", json=payload)
    assert response.status_code == 202
    body = response.json()

    usage_event = body["usage_event"]
    assert usage_event["project_id"] == "proj-123"
    assert usage_event["model_name"] == "demo-model"
    assert usage_event["baseline_latency_ms"] == 120.0
    assert usage_event["optimized_latency_ms"] == 80.0

    providers = {entry["provider"] for entry in body["dispatch"]}
    assert providers == {"aws", "gcp", "azure"}

    statuses = {entry["status"] for entry in body["dispatch"]}
    assert statuses <= {"skipped", "deferred", "error", "ok"}


def test_usage_endpoint_accepts_precomputed_usage_event():
    meter = UsageMeter()
    baseline = _snapshot(100.0, 80.0, 1024)
    optimized = _snapshot(60.0, 120.0, 1024)
    usage_event = meter.record_optimization(
        project_id="proj-456",
        model_name="custom-model",
        baseline_snapshot=baseline,
        optimized_snapshot=optimized,
    )

    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/usage",
        json={
            "usage_event": usage_event.to_dict(),
            "providers": ["aws"],
            "meter_name": "agnitra_opt_hours",
        },
    )

    assert response.status_code == 202
    body = response.json()
    assert body["usage_event"]["project_id"] == "proj-456"
    assert body["dispatch"][0]["provider"] == "aws"
    assert body["dispatch"][0]["status"] in {"skipped", "deferred", "ok", "error"}

