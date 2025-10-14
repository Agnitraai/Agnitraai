import json

from starlette.testclient import TestClient

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

