"""Snapshot-style regression tests for the dashboard API responses."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest
from starlette.testclient import TestClient

from agnitra.dashboard import create_app

SNAPSHOT_PATH = Path(__file__).parent / "__snapshots__" / "dashboard_api_snapshot.json"


@pytest.mark.skipif(
    not SNAPSHOT_PATH.exists(),
    reason="Snapshot fixture missing; ensure repository ships with baseline snapshot.",
)
def test_dashboard_api_snapshot_regression() -> None:
    """Assert the dashboard API responses stay aligned with the recorded snapshot."""
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

    with TestClient(create_app()) as client:
        response = client.post(
            "/upload",
            data={"model_name": "TinyLlama", "notes": "Snapshot regression demo"},
            files={
                "model_file": ("model.pt", b"pretend-model", "application/octet-stream"),
                "hardware_file": (
                    "hardware.json",
                    json.dumps(
                        {
                            "cpu": {"model": "AMD EPYC 7742"},
                            "gpus": [
                                {"name": "NVIDIA A100", "memory": "80GB", "count": 2}
                            ],
                            "memory": "512GB",
                        }
                    ),
                    "application/json",
                ),
                "log_file": ("run.log", "INFO starting run", "text/plain"),
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
            },
            follow_redirects=True,
        )
        response.raise_for_status()

        summary = client.get("/api/summary").json()
        uploads = client.get("/api/uploads").json()
        artifacts = client.get("/api/kernel-artifacts").json()
        sdk_pack_bytes = client.get("/export/sdk-pack").content

    archive = zipfile.ZipFile(io.BytesIO(sdk_pack_bytes), mode="r")
    archive_members = [
        "logs/notes_snapshot.txt"
        if name.startswith("logs/notes_") and name.endswith(".txt")
        else name
        for name in sorted(archive.namelist())
    ]
    archive_summary = json.loads(archive.read("summary.json"))
    archive.close()

    id_map: Dict[str, str] = {}
    snapshot_actual = {
        "summary": summary,
        "uploads": _canonicalize_identifiers(uploads, id_map),
        "artifacts": _canonicalize_identifiers(artifacts, id_map),
        "archive": {"members": archive_members, "summary": archive_summary},
    }

    expected = json.loads(SNAPSHOT_PATH.read_text())
    assert snapshot_actual == expected


def _canonicalize_identifiers(
    payload: Any, id_map: Dict[str, str] | None = None
) -> Any:
    """Replace random asset identifiers with deterministic placeholders."""
    if id_map is None:
        id_map = {}

    def assign(raw: str) -> str:
        if raw not in id_map:
            id_map[raw] = f"asset_{len(id_map) + 1}"
        return id_map[raw]

    def transform(value: Any) -> Any:
        if isinstance(value, dict):
            transformed: Dict[str, Any] = {}
            for key, item in value.items():
                if key == "identifier" and isinstance(item, str):
                    transformed[key] = assign(item)
                elif key == "download_url" and isinstance(item, str):
                    prefix, _, suffix = item.rpartition("/")
                    if suffix:
                        transformed[key] = f"{prefix}/{assign(suffix)}"
                    else:
                        transformed[key] = item
                else:
                    transformed[key] = transform(item)
            return transformed
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, str):
            # Normalise note filenames to avoid UUID churn.
            note_match = re.match(r"(notes)_([0-9a-f]{8})\.txt", value)
            if note_match:
                return f"{note_match.group(1)}_snapshot.txt"
        return value

    return transform(payload)
