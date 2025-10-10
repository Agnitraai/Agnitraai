"""Web dashboard application for Agnitra artifacts and performance telemetry."""

from __future__ import annotations

import asyncio
import io
import json
import mimetypes
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from markupsafe import Markup

from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
)
from starlette.routing import Route
from starlette.templating import Jinja2Templates


@dataclass
class DashboardAsset:
    """Metadata describing an uploaded artifact exposed by the dashboard."""

    identifier: str
    name: str
    path: Path
    category: str
    description: Optional[str] = None
    media_type: Optional[str] = None
    size_bytes: int = 0


# Backwards compatibility export expected by tests/imports.
KernelArtifact = DashboardAsset


@dataclass
class DashboardData:
    """In-memory store for dashboard uploads and derived analytics."""

    model_name: Optional[str] = None
    model_asset: Optional[DashboardAsset] = None
    hardware_asset: Optional[DashboardAsset] = None
    telemetry_before_asset: Optional[DashboardAsset] = None
    telemetry_after_asset: Optional[DashboardAsset] = None
    usage_event_asset: Optional[DashboardAsset] = None
    log_assets: List[DashboardAsset] = field(default_factory=list)
    telemetry_before: Optional[Dict[str, Any]] = None
    telemetry_after: Optional[Dict[str, Any]] = None
    usage_event: Optional[Dict[str, Any]] = None
    asset_index: Dict[str, DashboardAsset] = field(default_factory=dict)
    kernel_artifacts: List[DashboardAsset] = field(default_factory=list)


def create_app(templates_dir: Optional[Path] = None) -> Starlette:
    """Create and configure the Starlette application for the dashboard."""
    storage_dir = Path(tempfile.mkdtemp(prefix="agnitra_dashboard_"))
    data = DashboardData()
    lock = asyncio.Lock()
    templates_path = templates_dir or Path(__file__).parent / "templates"
    templates = Jinja2Templates(directory=str(templates_path))
    templates.env.filters.setdefault(
        "tojson", lambda value, indent=2: Markup(json.dumps(value, indent=indent))
    )

    def _register_asset(asset: DashboardAsset) -> None:
        data.asset_index[asset.identifier] = asset

    def _replace_asset(attr: str, asset: DashboardAsset) -> None:
        existing = getattr(data, attr)
        if isinstance(existing, DashboardAsset):
            data.asset_index.pop(existing.identifier, None)
        setattr(data, attr, asset)
        _register_asset(asset)

    def _append_asset(attr: str, asset: DashboardAsset) -> None:
        collection = getattr(data, attr)
        collection.append(asset)
        _register_asset(asset)

    async def dashboard_view(request: Request) -> HTMLResponse:
        tab = request.query_params.get("tab", "overview")
        async with lock:
            summary = build_performance_summary(data)
            layer_stats = build_layer_stats(data)
            artifacts = [
                {
                    "identifier": artifact.identifier,
                    "name": artifact.name,
                    "description": artifact.description,
                    "download_url": f"/assets/{artifact.identifier}",
                    "size": _format_size(artifact.size_bytes),
                    "media_type": artifact.media_type,
                }
                for artifact in data.kernel_artifacts
            ]
            model_name = data.model_name
            uploads = build_upload_overview(data, summary)

        context = {
            "request": request,
            "tab": tab,
            "summary": summary,
            "layer_stats": layer_stats,
            "artifacts": artifacts,
            "model_name": model_name,
            "uploads": uploads,
        }
        return templates.TemplateResponse(request, "dashboard.html", context)

    async def upload_artifacts(request: Request) -> Response:
        form = await request.form()
        model_name = form.get("model_name")
        notes = form.get("notes")

        upload_fields = {
            "model_file": form.get("model_file"),
            "telemetry_before": form.get("telemetry_before"),
            "telemetry_after": form.get("telemetry_after"),
            "usage_event_file": form.get("usage_event_file"),
            "hardware_file": form.get("hardware_file"),
            "log_file": form.get("log_file"),
            "optimized_ir": form.get("optimized_ir"),
            "kernel_file": form.get("kernel_file"),
        }

        async with lock:
            if model_name:
                data.model_name = model_name.strip() or None

            if isinstance(notes, str) and notes.strip():
                note_asset = _write_text_asset(
                    storage_dir,
                    f"notes_{uuid.uuid4().hex[:8]}.txt",
                    notes.strip(),
                    category="note",
                    description="Analyst note",
                )
                _append_asset("log_assets", note_asset)

            model_file = _as_upload(upload_fields["model_file"])
            if model_file:
                model_asset = await _store_upload_asset(
                    storage_dir,
                    model_file,
                    prefix="model",
                    category="model",
                    description="Uploaded model",
                )
                _replace_asset("model_asset", model_asset)

            log_file = _as_upload(upload_fields["log_file"])
            if log_file:
                log_asset = await _store_upload_asset(
                    storage_dir,
                    log_file,
                    prefix="log",
                    category="log",
                    description="Runtime log",
                )
                _append_asset("log_assets", log_asset)

            hardware_file = _as_upload(upload_fields["hardware_file"])
            if hardware_file:
                hardware_asset = await _store_upload_asset(
                    storage_dir,
                    hardware_file,
                    prefix="hardware",
                    category="hardware",
                    description="Hardware inventory",
                )
                _replace_asset("hardware_asset", hardware_asset)

            telemetry_before = _as_upload(upload_fields["telemetry_before"])
            if telemetry_before:
                baseline_asset, parsed = await _store_json_upload(
                    storage_dir,
                    telemetry_before,
                    prefix="baseline",
                    category="telemetry",
                    description="Baseline telemetry",
                )
                data.telemetry_before = parsed
                _replace_asset("telemetry_before_asset", baseline_asset)
                _maybe_set_model_name(data, parsed)

            telemetry_after = _as_upload(upload_fields["telemetry_after"])
            if telemetry_after:
                optimized_asset, parsed = await _store_json_upload(
                    storage_dir,
                    telemetry_after,
                    prefix="optimized",
                    category="telemetry",
                    description="Optimized telemetry",
                )
                data.telemetry_after = parsed
                _replace_asset("telemetry_after_asset", optimized_asset)
                _maybe_set_model_name(data, parsed)

            usage_file = _as_upload(upload_fields["usage_event_file"])
            if usage_file:
                usage_asset, parsed = await _store_json_upload(
                    storage_dir,
                    usage_file,
                    prefix="usage",
                    category="usage",
                    description="Usage event",
                )
                data.usage_event = parsed
                _replace_asset("usage_event_asset", usage_asset)
                _maybe_set_model_name(data, parsed)

            for key, label in (
                ("optimized_ir", "Optimized IR"),
                ("kernel_file", "Kernel Artifact"),
            ):
                upload = _as_upload(upload_fields[key])
                if upload:
                    artifact_asset = await _store_upload_asset(
                        storage_dir,
                        upload,
                        prefix="artifact",
                        category="artifact",
                        description=label,
                    )
                    data.kernel_artifacts.append(artifact_asset)
                    _register_asset(artifact_asset)

        referer = request.headers.get("referer") or "/"
        return RedirectResponse(url=referer, status_code=303)

    async def api_summary(request: Request) -> JSONResponse:
        async with lock:
            summary = build_performance_summary(data)
        return JSONResponse(summary)

    async def api_model_analyzer(request: Request) -> JSONResponse:
        async with lock:
            layer_stats = build_layer_stats(data)
        return JSONResponse({"layers": layer_stats})

    async def api_kernel_artifacts(request: Request) -> JSONResponse:
        async with lock:
            artifacts = [
                {
                    "identifier": artifact.identifier,
                    "name": artifact.name,
                    "description": artifact.description,
                    "download_url": f"/assets/{artifact.identifier}",
                    "media_type": artifact.media_type,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in data.kernel_artifacts
            ]
        return JSONResponse({"artifacts": artifacts})

    async def api_uploads(request: Request) -> JSONResponse:
        async with lock:
            uploads = build_upload_overview(data)
        return JSONResponse(uploads)

    async def download_asset(request: Request) -> FileResponse:
        artifact_id = request.path_params.get("artifact_id")
        asset_id = request.path_params.get("asset_id") or artifact_id
        if asset_id is None:
            raise HTTPException(status_code=404, detail="Unknown asset")
        async with lock:
            asset = data.asset_index.get(asset_id)
            if not asset:
                raise HTTPException(status_code=404, detail="Artifact not found")
            path = asset.path
            filename = asset.name
            media_type = asset.media_type or "application/octet-stream"

        return FileResponse(
            path,
            media_type=media_type,
            filename=filename,
        )

    async def export_sdk_pack(request: Request) -> Response:
        async with lock:
            if not data.telemetry_before and not data.telemetry_after:
                raise HTTPException(
                    status_code=400, detail="No telemetry available for export"
                )
            archive_bytes = build_sdk_pack_bytes(data)

        headers = {
            "Content-Disposition": "attachment; filename=agnitra_sdk_pack.zip"
        }
        return Response(
            content=archive_bytes,
            media_type="application/zip",
            headers=headers,
        )

    routes = [
        Route("/", dashboard_view, methods=["GET"]),
        Route("/upload", upload_artifacts, methods=["POST"]),
        Route("/api/summary", api_summary, methods=["GET"]),
        Route("/api/model-analyzer", api_model_analyzer, methods=["GET"]),
        Route("/api/kernel-artifacts", api_kernel_artifacts, methods=["GET"]),
        Route("/api/uploads", api_uploads, methods=["GET"]),
        Route("/assets/{asset_id}", download_asset, methods=["GET"]),
        Route("/artifacts/{artifact_id}", download_asset, methods=["GET"]),
        Route("/export/sdk-pack", export_sdk_pack, methods=["GET"]),
    ]

    app = Starlette(
        debug=False,
        routes=routes,
    )
    app.state.storage_dir = storage_dir
    app.state.dashboard_data = data
    app.state.lock = lock
    app.state.templates = templates
    return app


def _safe_filename(filename: Optional[str]) -> str:
    if not filename:
        return "unnamed"
    return Path(filename).name


def _as_upload(value: Any) -> Optional[UploadFile]:
    return value if isinstance(value, UploadFile) else None


async def _persist_upload(
    storage_dir: Path, upload: UploadFile, prefix: str
) -> Tuple[Path, bytes, str]:
    safe_name = _safe_filename(upload.filename)
    destination = storage_dir / f"{prefix}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    destination.write_bytes(content)
    return destination, content, safe_name


async def _store_json_upload(
    storage_dir: Path,
    upload: UploadFile,
    prefix: str,
    *,
    category: str,
    description: Optional[str] = None,
) -> Tuple[DashboardAsset, Dict[str, Any]]:
    destination, content, safe_name = await _persist_upload(storage_dir, upload, prefix)
    try:
        parsed = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON payload for {safe_name}: {exc}",
        ) from exc
    asset = DashboardAsset(
        identifier=uuid.uuid4().hex,
        name=safe_name,
        path=destination,
        category=category,
        description=description,
        media_type=_guess_media_type(upload, safe_name),
        size_bytes=len(content),
    )
    return asset, parsed


async def _store_upload_asset(
    storage_dir: Path,
    upload: UploadFile,
    *,
    prefix: str,
    category: str,
    description: Optional[str] = None,
) -> DashboardAsset:
    destination, content, safe_name = await _persist_upload(storage_dir, upload, prefix)
    return DashboardAsset(
        identifier=uuid.uuid4().hex,
        name=safe_name,
        path=destination,
        category=category,
        description=description,
        media_type=_guess_media_type(upload, safe_name),
        size_bytes=len(content),
    )


def _write_text_asset(
    storage_dir: Path,
    filename: str,
    text: str,
    *,
    category: str,
    description: Optional[str] = None,
) -> DashboardAsset:
    path = storage_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    path.write_bytes(encoded)
    return DashboardAsset(
        identifier=uuid.uuid4().hex,
        name=filename,
        path=path,
        category=category,
        description=description,
        media_type="text/plain",
        size_bytes=len(encoded),
    )


def _guess_media_type(upload: UploadFile, filename: str) -> Optional[str]:
    if upload.content_type and upload.content_type != "application/octet-stream":
        return upload.content_type
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or upload.content_type


def build_upload_overview(
    data: DashboardData, summary: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Prepare a view model describing uploaded assets for rendering or APIs."""
    summary = summary or build_performance_summary(data)
    uploads = {
        "model": _serialize_asset(data.model_asset),
        "hardware": _build_hardware_entry(data.hardware_asset),
        "telemetry": {
            "baseline": _build_telemetry_entry(
                "baseline",
                data.telemetry_before_asset,
                data.telemetry_before,
                summary,
            ),
            "optimized": _build_telemetry_entry(
                "optimized",
                data.telemetry_after_asset,
                data.telemetry_after,
                summary,
            ),
        },
        "usage": _build_usage_entry(data.usage_event_asset, data.usage_event),
        "logs": [_build_log_entry(asset) for asset in data.log_assets],
    }
    return uploads


def _serialize_asset(
    asset: Optional[DashboardAsset],
    *,
    preview: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if asset is None:
        return None
    descriptor: Dict[str, Any] = {
        "identifier": asset.identifier,
        "name": asset.name,
        "category": asset.category,
        "description": asset.description,
        "download_url": f"/assets/{asset.identifier}",
        "size": _format_size(asset.size_bytes),
        "size_bytes": asset.size_bytes,
        "media_type": asset.media_type,
    }
    if preview:
        descriptor["preview"] = preview
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            descriptor[key] = value
    return descriptor


def _build_log_entry(asset: DashboardAsset) -> Dict[str, Any]:
    preview = _read_text_preview(asset.path)
    extra = {"is_note": asset.category == "note"}
    entry = _serialize_asset(asset, preview=preview, extra=extra)
    # _serialize_asset only returns None when asset is None, which never happens here.
    assert entry is not None
    return entry


def _build_hardware_entry(
    asset: Optional[DashboardAsset],
) -> Optional[Dict[str, Any]]:
    if asset is None:
        return None
    payload = None
    preview = None
    try:
        payload = json.loads(asset.path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        preview = _read_text_preview(asset.path)
    else:
        preview = _json_preview(payload)
    extra: Dict[str, Any] = {}
    if payload is not None:
        summary = _summarize_hardware_payload(payload)
        if summary:
            extra["summary"] = summary
    return _serialize_asset(asset, preview=preview, extra=extra or None)


def _build_telemetry_entry(
    variant: str,
    asset: Optional[DashboardAsset],
    payload: Optional[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if asset is None:
        return None
    metrics: List[Dict[str, Any]] = []
    for metric in summary.get("metrics", []):
        value = metric.get("baseline" if variant == "baseline" else "optimized")
        if value is not None:
            metrics.append({"label": metric["label"], "value": value})
    events = _get_events(payload)
    extra: Dict[str, Any] = {
        "variant": variant,
        "event_count": len(events) if events is not None else 0,
    }
    if metrics:
        extra["metrics"] = metrics
    if payload and isinstance(payload.get("bottleneck"), dict):
        extra["bottleneck"] = _summarize_bottleneck(payload["bottleneck"])
    preview = _json_preview(payload)
    return _serialize_asset(asset, preview=preview, extra=extra)


def _build_usage_entry(
    asset: Optional[DashboardAsset],
    payload: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if asset is None:
        return None
    highlights: List[Dict[str, Any]] = []
    if payload:
        for dotted_key, label in (
            ("baseline_latency_ms", "Baseline latency (ms)"),
            ("optimized_latency_ms", "Optimized latency (ms)"),
            ("baseline_tokens_per_sec", "Baseline tokens/s"),
            ("optimized_tokens_per_sec", "Optimized tokens/s"),
            ("gpu_hours_saved", "GPU hours saved"),
            ("cost_savings", "Cost savings (USD)"),
            ("total_billable", "Total billable (USD)"),
        ):
            value = _dig(payload, dotted_key)
            if value is not None:
                highlights.append({"label": label, "value": value})
    extra = {"highlights": highlights} if highlights else {}
    preview = _json_preview(payload)
    return _serialize_asset(asset, preview=preview, extra=extra or None)


def _summarize_bottleneck(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for key in ("op", "node", "name", "layer"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            summary["name"] = value
            break
    latency = payload.get("latency_ms") or payload.get("duration_ms")
    if latency is not None:
        summary["latency_ms"] = _to_float(latency)
    shape = payload.get("shape")
    if shape:
        summary["shape"] = shape
    return summary


def _summarize_hardware_payload(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        summary: Dict[str, Any] = {}
        gpu_section = payload.get("gpus") or payload.get("devices") or payload.get(
            "gpu"
        )
        gpus: List[Dict[str, Any]] = []
        if isinstance(gpu_section, list):
            for item in gpu_section[:4]:
                parsed = _summarize_gpu_entry(item)
                if parsed:
                    gpus.append(parsed)
        elif isinstance(gpu_section, dict):
            parsed = _summarize_gpu_entry(gpu_section)
            if parsed:
                gpus.append(parsed)
        if gpus:
            summary["gpus"] = gpus
            summary["gpu_count"] = (
                len(gpu_section)
                if isinstance(gpu_section, list)
                else summary.get("gpus") and len(summary["gpus"])
            )
        cpu_section = payload.get("cpu") or payload.get("cpus")
        if isinstance(cpu_section, dict):
            cpu_name = cpu_section.get("model") or cpu_section.get("name")
            if isinstance(cpu_name, str):
                summary["cpu"] = cpu_name
        memory = (
            payload.get("memory")
            or payload.get("system_memory")
            or payload.get("ram")
        )
        if memory is not None:
            summary["memory"] = memory
        return summary or None
    if isinstance(payload, list):
        return {"entries": len(payload)}
    return None


def _summarize_gpu_entry(payload: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    summary: Dict[str, Any] = {}
    name = payload.get("name") or payload.get("model")
    if isinstance(name, str):
        summary["name"] = name
    for mem_key in ("memory", "memory_gb", "total_memory"):
        value = payload.get(mem_key)
        if value is not None:
            summary["memory"] = value
            break
    count = payload.get("count") or payload.get("quantity")
    if isinstance(count, int):
        summary["count"] = count
    return summary or None


def _read_text_preview(path: Path, limit: int = 700) -> Optional[str]:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            snippet = handle.read(limit + 1)
    except OSError:
        return None
    snippet = snippet.strip()
    if not snippet:
        return None
    if len(snippet) > limit:
        return snippet[:limit].rstrip() + "…"
    return snippet


def _json_preview(payload: Optional[Dict[str, Any]], limit: int = 700) -> Optional[str]:
    if not payload:
        return None
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if len(serialized) > limit:
        return serialized[:limit].rstrip() + "…"
    return serialized


def _format_size(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size_bytes} B"


def _maybe_set_model_name(data: DashboardData, payload: Dict[str, Any]) -> None:
    name = payload.get("model_name") or payload.get("model")
    if isinstance(name, str) and name.strip():
        data.model_name = name.strip()


def build_performance_summary(data: DashboardData) -> Dict[str, Any]:
    """Compute aggregate metrics for the overview and benchmarks tabs."""

    baseline = data.telemetry_before or {}
    optimized = data.telemetry_after or {}
    usage = data.usage_event or {}

    summary: Dict[str, Any] = {
        "model_name": data.model_name,
        "metrics": [],
    }

    metrics_definitions = {
        "latency_ms": {
            "label": "Latency (ms)",
            "baseline_keys": [
                "latency_ms",
                "metrics.latency_ms",
                "bottleneck.latency_ms",
                "baseline_latency_ms",
            ],
            "optimized_keys": [
                "latency_ms",
                "metrics.latency_ms",
                "bottleneck.latency_ms",
                "optimized_latency_ms",
            ],
        },
        "tokens_per_sec": {
            "label": "Tokens / second",
            "baseline_keys": [
                "tokens_per_sec",
                "throughput.tokens_per_sec",
                "baseline_tokens_per_sec",
            ],
            "optimized_keys": [
                "tokens_per_sec",
                "throughput.tokens_per_sec",
                "optimized_tokens_per_sec",
            ],
        },
        "gpu_utilisation": {
            "label": "GPU utilisation",
            "baseline_keys": [
                "gpu.gpu_utilisation",
                "gpu.utilisation",
                "behavior.gpu_util_mean",
                "baseline_gpu_utilisation",
            ],
            "optimized_keys": [
                "gpu.gpu_utilisation",
                "gpu.utilisation",
                "behavior.gpu_util_mean",
                "optimized_gpu_utilisation",
            ],
        },
        "gpu_hours": {
            "label": "GPU hours",
            "baseline_keys": [
                "gpu_hours",
                "gpu_hours_before",
                "usage.gpu_hours_before",
            ],
            "optimized_keys": [
                "gpu_hours",
                "gpu_hours_after",
                "usage.gpu_hours_after",
            ],
        },
        "cost": {
            "label": "Cost (USD)",
            "baseline_keys": ["cost_before", "usage.cost_before"],
            "optimized_keys": ["cost_after", "usage.cost_after"],
        },
    }

    metrics = []
    for key, definition in metrics_definitions.items():
        baseline_value = _resolve_metric(
            baseline, optimized, usage, definition["baseline_keys"], variant="baseline"
        )
        optimized_value = _resolve_metric(
            optimized, baseline, usage, definition["optimized_keys"], variant="optimized"
        )
        metrics.append(
            _build_metric_entry(
                label=definition["label"],
                baseline_value=baseline_value,
                optimized_value=optimized_value,
            )
        )

    metrics = [
        metric
        for metric in metrics
        if metric["baseline"] is not None or metric["optimized"] is not None
    ]
    summary["metrics"] = metrics
    return summary


def _resolve_metric(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    usage: Dict[str, Any],
    candidates: Iterable[str],
    variant: str,
) -> Optional[float]:
    """Find the first numeric metric across several sources."""
    for source in (primary, secondary, usage):
        if not source:
            continue
        for key in candidates:
            value = _dig(source, key)
            if value is not None:
                return _to_float(value)
        tagged_key = f"{variant}_{candidates[0]}"
        value = _dig(source, tagged_key)
        if value is not None:
            return _to_float(value)
    return None


def _dig(source: Dict[str, Any], dotted_path: str) -> Optional[Any]:
    """Follow dot-separated keys inside nested dicts."""
    current: Any = source
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_metric_entry(
    label: str,
    baseline_value: Optional[float],
    optimized_value: Optional[float],
) -> Dict[str, Any]:
    change = None
    change_pct = None
    if baseline_value is not None and optimized_value is not None:
        change = optimized_value - baseline_value
        if baseline_value:
            change_pct = (optimized_value / baseline_value - 1) * 100
    return {
        "label": label,
        "baseline": baseline_value,
        "optimized": optimized_value,
        "delta": change,
        "delta_pct": change_pct,
    }


def build_layer_stats(data: DashboardData) -> List[Dict[str, Any]]:
    """Merge baseline and optimized layer telemetry into comparable rows."""
    layer_map: Dict[str, Dict[str, Any]] = {}

    def ingest(events: Optional[Iterable[Dict[str, Any]]], variant: str) -> None:
        if not events:
            return
        for raw in events:
            if not isinstance(raw, dict):
                continue
            name = str(
                raw.get("name")
                or raw.get("op")
                or raw.get("layer")
                or raw.get("node")
                or f"{variant}-layer-{len(layer_map)+1}"
            )
            entry = layer_map.setdefault(
                name,
                {
                    "name": name,
                    "baseline_latency_ms": None,
                    "optimized_latency_ms": None,
                    "metadata": {},
                },
            )
            latency = _to_float(
                raw.get("latency_ms")
                or raw.get("duration_ms")
                or raw.get("latency")
            )
            if latency is not None:
                key = (
                    "baseline_latency_ms"
                    if variant == "baseline"
                    else "optimized_latency_ms"
                )
                entry[key] = latency

            extra = {
                k: v
                for k, v in raw.items()
                if k
                not in {
                    "name",
                    "op",
                    "layer",
                    "node",
                    "latency_ms",
                    "latency",
                    "duration_ms",
                }
            }
            entry["metadata"].update(extra)

    ingest(_get_events(data.telemetry_before), "baseline")
    ingest(_get_events(data.telemetry_after), "optimized")
    return list(layer_map.values())


def _get_events(payload: Optional[Dict[str, Any]]) -> Optional[Iterable[Dict[str, Any]]]:
    if not payload:
        return None
    events = payload.get("events")
    if isinstance(events, list):
        return events
    return None


def build_sdk_pack_bytes(data: DashboardData) -> bytes:
    """Create an exportable ZIP archive with telemetry and kernel artifacts."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        summary = build_performance_summary(data)
        zf.writestr("summary.json", json.dumps(summary, indent=2))

        if data.telemetry_before:
            zf.writestr(
                "telemetry/baseline.json",
                json.dumps(data.telemetry_before, indent=2),
            )
        if data.telemetry_after:
            zf.writestr(
                "telemetry/optimized.json",
                json.dumps(data.telemetry_after, indent=2),
            )
        if data.usage_event:
            zf.writestr(
                "telemetry/usage_event.json",
                json.dumps(data.usage_event, indent=2),
            )
        if data.hardware_asset and data.hardware_asset.path.exists():
            zf.write(
                data.hardware_asset.path,
                arcname=f"context/{data.hardware_asset.name}",
            )
        for log_asset in data.log_assets:
            if log_asset.path.exists():
                arcname = f"logs/{log_asset.name}"
                zf.write(log_asset.path, arcname=arcname)
        for artifact in data.kernel_artifacts:
            if artifact.path.exists():
                arcname = f"artifacts/{artifact.name}"
                zf.write(artifact.path, arcname=arcname)

    buffer.seek(0)
    return buffer.read()


__all__ = [
    "DashboardAsset",
    "DashboardData",
    "KernelArtifact",
    "build_layer_stats",
    "build_performance_summary",
    "build_upload_overview",
    "build_sdk_pack_bytes",
    "create_app",
]
