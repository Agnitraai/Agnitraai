"""Starlette application exposing the Agentic Optimization and Marketplace APIs."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import json
import os
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from starlette.applications import Starlette
from starlette.datastructures import FormData, UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agnitra.api.marketplace import (
    MarketplaceDispatcher,
    create_default_dispatcher,
    usage_event_to_record,
)
from agnitra.api.auth import ApiKeyAuthenticator
from agnitra.api.billing import StripeBillingClient
from agnitra.api.queue import OptimizationQueue
from agnitra.core.metering import UsageEvent, UsageMeter
from agnitra.core.runtime import OptimizationSnapshot

from .service import run_agentic_optimization

AUTHENTICATOR = ApiKeyAuthenticator.from_env()
USAGE_METER = UsageMeter()
STRIPE_CLIENT = StripeBillingClient.from_env()


async def _queue_worker(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def _run():
        return run_agentic_optimization(
            payload.get("model_graph"),
            payload.get("telemetry"),
            payload.get("target", ""),
            project_id=payload.get("project_id", "default"),
            model_name=payload.get("model_name"),
            usage_meter=USAGE_METER,
            tokens_processed=payload.get("tokens_processed"),
            stripe_client=STRIPE_CLIENT,
            customer_id=payload.get("customer_id"),
            meter_metadata=payload.get("meter_metadata"),
        )

    if hasattr(asyncio, "to_thread"):
        return await asyncio.to_thread(_run)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run)


ASYNC_QUEUE = OptimizationQueue(
    _queue_worker,
    concurrency=max(1, int(os.environ.get("AGNITRA_API_WORKERS", "2"))),
)


async def _healthcheck(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def _optimize(request: Request) -> Response:
    _require_api_key(request)
    if _is_json_request(request):
        payload = await request.json()
        return await _handle_json_payload(payload, request)

    form = await request.form()
    return await _handle_form_payload(form, request)


def create_app() -> Starlette:
    """Return a configured Starlette application."""

    routes = [
        Route("/health", _healthcheck, methods=["GET"]),
        Route("/optimize", _optimize, methods=["POST"]),
        Route("/usage", _usage, methods=["POST"]),
        Route("/jobs/{job_id}", _job_status, methods=["GET"]),
    ]
    return Starlette(debug=False, routes=routes)


async def _handle_json_payload(payload: Any, request: Request) -> Response:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object.")

    target = _extract_target(payload)
    model_graph = payload.get("model_graph")
    telemetry = payload.get("telemetry")
    project_id = _coerce_project_id(payload.get("project_id"))
    model_name = _coerce_optional_str(payload.get("model_name"))
    tokens_processed = _coerce_optional_int(payload.get("tokens_processed"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None
    customer_id = _coerce_customer_id(request, payload)

    if _requires_async(payload):
        return await _submit_async_job(
            model_graph=model_graph,
            telemetry=telemetry,
            target=target,
            project_id=project_id,
            model_name=model_name,
            tokens_processed=tokens_processed,
            customer_id=customer_id,
            meter_metadata=metadata,
        )

    try:
        result = run_agentic_optimization(
            model_graph,
            telemetry,
            target,
            project_id=project_id,
            model_name=model_name,
            usage_meter=USAGE_METER,
            tokens_processed=tokens_processed,
            stripe_client=STRIPE_CLIENT,
            customer_id=customer_id,
            meter_metadata=metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Optimization failed") from exc

    return JSONResponse(result)


async def _handle_form_payload(form: FormData, request: Request) -> Response:
    target = _extract_target(form)

    model_upload = _first_upload(form, ("model_graph", "model_graph.json"))
    if model_upload is None:
        raise HTTPException(status_code=400, detail="model_graph upload is required.")

    telemetry_upload = _first_upload(form, ("telemetry", "telemetry.json"))

    model_graph = await _read_json_upload(model_upload, "model_graph")
    telemetry = await _read_json_upload(telemetry_upload, "telemetry") if telemetry_upload else {}

    project_id = _coerce_project_id(form.get("project_id"))
    model_name = _coerce_optional_str(form.get("model_name"))
    tokens_processed = _coerce_optional_int(form.get("tokens_processed"))
    customer_id = _coerce_customer_id(request, form)
    metadata = _parse_metadata_field(form.get("metadata"))

    if _requires_async(form):
        return await _submit_async_job(
            model_graph=model_graph,
            telemetry=telemetry,
            target=target,
            project_id=project_id,
            model_name=model_name,
            tokens_processed=tokens_processed,
            customer_id=customer_id,
            meter_metadata=metadata,
        )

    try:
        result = run_agentic_optimization(
            model_graph,
            telemetry,
            target,
            project_id=project_id,
            model_name=model_name,
            usage_meter=USAGE_METER,
            tokens_processed=tokens_processed,
            stripe_client=STRIPE_CLIENT,
            customer_id=customer_id,
            meter_metadata=metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail="Optimization failed") from exc

    return JSONResponse(result)


def _is_json_request(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    media_type = content_type.split(";", 1)[0].strip().lower()
    return media_type == "application/json"


def _require_api_key(request: Request) -> None:
    if not AUTHENTICATOR.accepted_digests:
        return
    api_key = _extract_api_key(request)
    if not AUTHENTICATOR.is_valid(api_key):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _extract_api_key(request: Request) -> Optional[str]:
    header = request.headers.get("x-api-key")
    if isinstance(header, str) and header.strip():
        return header.strip()
    auth_header = request.headers.get("authorization")
    if isinstance(auth_header, str):
        parts = auth_header.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip()
    return None


def _requires_async(source: Any) -> bool:
    if isinstance(source, Mapping):
        value = source.get("async") or source.get("queue") or source.get("background")
        mode = source.get("mode")
        if isinstance(value, (str, bytes)):
            if str(value).strip().lower() in {"1", "true", "yes"}:
                return True
        if isinstance(value, bool) and value:
            return True
        if isinstance(mode, str) and mode.strip().lower() == "async":
            return True
    if isinstance(source, FormData):
        value = source.get("async") or source.get("queue") or source.get("background")
        if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}:
            return True
    return False


def _coerce_project_id(value: Any) -> str:
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
        text = text.strip()
        if text:
            return text
    return "default"


def _coerce_optional_str(value: Any) -> Optional[str]:
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
        text = text.strip()
        if text:
            return text
    return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_metadata_field(value: Any) -> Optional[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="ignore") if isinstance(value, bytes) else value
        text = text.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="metadata field must contain valid JSON object.")
        if isinstance(parsed, Mapping):
            return parsed
        raise HTTPException(status_code=400, detail="metadata JSON must be an object.")
    return None


async def _submit_async_job(**payload: Any) -> Response:
    if payload.get("model_graph") is None:
        raise HTTPException(status_code=400, detail="model_graph is required.")
    job = await ASYNC_QUEUE.enqueue(dict(payload))
    return JSONResponse({"status": "queued", "job_id": job.identifier}, status_code=202)


def _coerce_customer_id(request: Request, source: Any) -> Optional[str]:
    header = request.headers.get("x-customer-id") or request.headers.get("agnitra-customer-id")
    if isinstance(header, str) and header.strip():
        return header.strip()
    if isinstance(source, Mapping):
        return _coerce_optional_str(source.get("customer_id"))
    if isinstance(source, FormData):
        return _coerce_optional_str(source.get("customer_id"))
    return None


async def _job_status(request: Request) -> Response:
    _require_api_key(request)
    job_id = request.path_params.get("job_id")
    if not isinstance(job_id, str):
        raise HTTPException(status_code=400, detail="Missing job identifier.")
    job = ASYNC_QUEUE.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    payload = {
        "job_id": job.identifier,
        "status": job.status,
    }
    if job.result is not None and job.status == "completed":
        payload["result"] = job.result
    if job.error:
        payload["error"] = job.error
    return JSONResponse(payload)


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


_MARKETPLACE_DISPATCHER: MarketplaceDispatcher = create_default_dispatcher()


async def _usage(request: Request) -> Response:
    _require_api_key(request)
    if not _is_json_request(request):
        raise HTTPException(status_code=415, detail="Usage endpoint requires application/json.")

    payload = await request.json()
    if not isinstance(payload, Mapping):
        raise HTTPException(status_code=400, detail="Usage request must be a JSON object.")

    usage_event_payload = payload.get("usage_event")
    if isinstance(usage_event_payload, Mapping):
        usage_event = _usage_event_from_mapping(usage_event_payload)
    else:
        usage_event = _build_usage_event(payload)

    meter_name = _coerce_meter_name(payload.get("meter_name"))
    quantity_field = payload.get("quantity_field")
    providers = _coerce_providers(payload.get("providers"))

    record = usage_event_to_record(
        usage_event,
        meter_name=meter_name,
        quantity_field=quantity_field if isinstance(quantity_field, str) else "gpu_hours_after",
    )

    dispatch_results = _MARKETPLACE_DISPATCHER.dispatch(record, providers=providers)

    return JSONResponse(
        {
            "status": "accepted",
            "usage_event": usage_event.to_dict(),
            "dispatch": [
                {
                    "provider": result.provider,
                    "status": result.status,
                    "detail": result.detail,
                    "payload": result.payload,
                }
                for result in dispatch_results
            ],
        },
        status_code=202,
    )


def _coerce_meter_name(value: Any) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "runtime_optimization_hours"


def _coerce_providers(value: Any) -> Optional[Sequence[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        providers = []
        for item in value:
            if isinstance(item, (str, bytes)):
                providers.append(item.decode("utf-8") if isinstance(item, bytes) else item)
        return providers or None
    return None


def _usage_event_from_mapping(payload: Mapping[str, Any]) -> UsageEvent:
    required_fields = {field.name for field in dataclasses.fields(UsageEvent)}
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise HTTPException(status_code=400, detail=f"usage_event missing fields: {', '.join(missing)}")

    normalised: dict[str, Any] = {}
    for field in dataclasses.fields(UsageEvent):
        name = field.name
        value = payload[name]
        if name == "timestamp":
            value = _parse_timestamp(value)
        elif name in {"gpu_util_before", "gpu_util_after"} and value is not None:
            value = _safe_float(value)
        elif name in _FLOAT_USAGE_FIELDS:
            value = _safe_float(value)
        elif name == "tokens_processed":
            value = _safe_int(value)
        elif name == "metadata":
            if isinstance(value, Mapping):
                value = dict(value)
            else:
                value = {}
        elif name in {"project_id", "model_name", "currency"}:
            if not isinstance(value, str) or not value.strip():
                raise HTTPException(status_code=400, detail=f"{name} must be a non-empty string.")
            value = value.strip()
        normalised[name] = value

    try:
        return UsageEvent(**normalised)
    except TypeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _build_usage_event(payload: Mapping[str, Any]) -> UsageEvent:
    project_id = _coerce_string(payload.get("project_id"))
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required.")

    model_name = _coerce_string(payload.get("model_name")) or "unknown_model"

    baseline_payload = payload.get("baseline")
    optimized_payload = payload.get("optimized")
    if not isinstance(baseline_payload, Mapping) or not isinstance(optimized_payload, Mapping):
        raise HTTPException(status_code=400, detail="baseline and optimized sections are required.")

    baseline_snapshot = _snapshot_from_mapping(baseline_payload)
    optimized_snapshot = _snapshot_from_mapping(optimized_payload)

    meter = UsageMeter(
        rate_per_gpu_hour=_safe_float(payload.get("rate_per_gpu_hour"), 2.5),
        margin_pct=_safe_float(payload.get("success_margin_pct", payload.get("margin_pct")), 0.2),
        currency=_coerce_string(payload.get("currency")) or "USD",
    )

    tokens_processed = payload.get("tokens_processed")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}

    return meter.record_optimization(
        project_id=project_id,
        model_name=model_name,
        baseline_snapshot=baseline_snapshot,
        optimized_snapshot=optimized_snapshot,
        tokens_processed=_safe_int(tokens_processed) if tokens_processed is not None else None,
        metadata=dict(metadata),
    )


def _snapshot_from_mapping(payload: Mapping[str, Any]) -> OptimizationSnapshot:
    latency = _safe_float(payload.get("latency_ms"))
    tokens_per_sec = max(_safe_float(payload.get("tokens_per_sec"), default=0.0), 1e-6)
    tokens_processed = _safe_int(payload.get("tokens_processed"), default=0)
    gpu_util = payload.get("gpu_utilization")
    if gpu_util is not None:
        gpu_util = _safe_float(gpu_util)

    telemetry = payload.get("telemetry")
    if isinstance(telemetry, Mapping):
        telemetry_data: dict[str, Any] = dict(telemetry)
    else:
        telemetry_data = {}

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        metadata_map: dict[str, Any] = dict(metadata)
    else:
        metadata_map = {}

    return OptimizationSnapshot(
        latency_ms=latency,
        tokens_per_sec=tokens_per_sec,
        tokens_processed=tokens_processed,
        gpu_utilization=gpu_util,
        telemetry=telemetry_data,
        metadata=metadata_map,
    )


def _coerce_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return ""


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_timestamp(value: Any) -> _dt.datetime:
    if isinstance(value, (int, float)):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.timezone.utc)
    if isinstance(value, str):
        candidates = [value, value.replace("Z", "+00:00")]
        for candidate in candidates:
            try:
                parsed = _dt.datetime.fromisoformat(candidate)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.timezone.utc)
            return parsed
        raise HTTPException(status_code=400, detail="timestamp must be an ISO 8601 string.")
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_dt.timezone.utc)
        return value
    raise HTTPException(status_code=400, detail="timestamp uses an unsupported type.")


_FLOAT_USAGE_FIELDS = {
    "baseline_latency_ms",
    "optimized_latency_ms",
    "baseline_tokens_per_sec",
    "optimized_tokens_per_sec",
    "gpu_hours_before",
    "gpu_hours_after",
    "gpu_hours_saved",
    "performance_uplift_pct",
    "cost_before",
    "cost_after",
    "cost_savings",
    "usage_charge",
    "success_fee",
    "total_billable",
}
