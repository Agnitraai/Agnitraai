"""Starlette application exposing the Agentic Optimization API."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional, Sequence

from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .service import run_agentic_optimization


async def _healthcheck(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def _optimize(request: Request) -> Response:
    if _is_json_request(request):
        payload = await request.json()
        return await _handle_json_payload(payload)

    form = await request.form()
    return await _handle_form_payload(form)


def create_app() -> Starlette:
    """Return a configured Starlette application."""

    routes = [
        Route("/health", _healthcheck, methods=["GET"]),
        Route("/optimize", _optimize, methods=["POST"]),
    ]
    return Starlette(debug=False, routes=routes)


async def _handle_json_payload(payload: Any) -> Response:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")

    target = _extract_target(payload)
    model_graph = payload.get("model_graph")
    telemetry = payload.get("telemetry")

    try:
        result = run_agentic_optimization(model_graph, telemetry, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Optimization failed") from exc

    return JSONResponse(result)


async def _handle_form_payload(form: FormData) -> Response:
    target = _extract_target(form)

    model_upload = _first_upload(form, ("model_graph", "model_graph.json"))
    if model_upload is None:
        raise HTTPException(status_code=400, detail="model_graph upload is required.")

    telemetry_upload = _first_upload(form, ("telemetry", "telemetry.json"))

    model_graph = await _read_json_upload(model_upload, "model_graph")
    telemetry = await _read_json_upload(telemetry_upload, "telemetry") if telemetry_upload else {}

    try:
        result = run_agentic_optimization(model_graph, telemetry, target)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Optimization failed") from exc

    return JSONResponse(result)


def _is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json"


def _extract_target(source: Any) -> str:
    if isinstance(source, (dict, FormData)):
        for key in ("target", "device", "accelerator", "hardware"):
            value = source.get(key)  # type: ignore[arg-type]
            if isinstance(value, (str, bytes)):
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="ignore")
                value = value.strip()
                if value:
                    return value
        raise HTTPException(status_code=400, detail="target field is required.")
    if isinstance(source, Mapping):
        value = source.get("target")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _first_upload(form: FormData, candidates: Sequence[str]) -> Optional[UploadFile]:
    for key in candidates:
        value = form.get(key)  # type: ignore[arg-type]
        if isinstance(value, UploadFile):
            return value
    return None


async def _read_json_upload(upload: UploadFile, label: str) -> Any:
    if upload is None:
        return {}
    try:
        raw = await upload.read()
    finally:
        await upload.close()
    if not raw:
        raise HTTPException(status_code=400, detail=f"{label} payload is empty.")
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must contain valid JSON.") from exc
