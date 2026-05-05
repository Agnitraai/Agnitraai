from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_targets_runtime_service() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "uvicorn" in dockerfile
    assert "agnitra.api.app:create_app" in dockerfile
    assert "EXPOSE 8080" in dockerfile


def test_optimize_routes_are_exposed() -> None:
    app_module = (PROJECT_ROOT / "agnitra" / "api" / "app.py").read_text(encoding="utf-8")
    assert 'Route("/optimize"' in app_module
    assert 'Route("/jobs/{job_id}"' in app_module
    assert 'WebSocketRoute("/ws/jobs/{job_id}"' in app_module
