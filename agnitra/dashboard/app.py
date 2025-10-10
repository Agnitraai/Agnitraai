"""Web dashboard application for Agnitra artifacts and performance telemetry."""

from __future__ import annotations

import asyncio
import io
import json
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
class KernelArtifact:
    """Metadata describing an uploaded kernel or IR artifact."""

    identifier: str
    name: str
    path: Path
    description: Optional[str] = None


@dataclass
class DashboardData:
    """In-memory store for dashboard uploads and derived analytics."""

    model_name: Optional[str] = None
    model_path: Optional[Path] = None
    hardware_path: Optional[Path] = None
    logs: List[Path] = field(default_factory=list)
    telemetry_before: Optional[Dict[str, Any]] = None
    telemetry_after: Optional[Dict[str, Any]] = None
    usage_event: Optional[Dict[str, Any]] = None
    usage_event_path: Optional[Path] = None
    kernel_artifacts: List[KernelArtifact] = field(default_factory=list)


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
                    "download_url": f"/artifacts/{artifact.identifier}",
                }
                for artifact in data.kernel_artifacts
            ]
            model_name = data.model_name

        context = {
            "request": request,
            "tab": tab,
            "summary": summary,
            "layer_stats": layer_stats,
            "artifacts": artifacts,
            "model_name": model_name,
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
                notes_path = _write_text_file(
                    storage_dir,
                    f"notes_{uuid.uuid4().hex[:8]}.txt",
                    notes.strip(),
                )
                data.logs.append(notes_path)

            model_file = _as_upload(upload_fields["model_file"])
            if model_file:
                data.model_path = await _store_upload(
                    storage_dir, model_file, prefix="model"
                )

            log_file = _as_upload(upload_fields["log_file"])
            if log_file:
                data.logs.append(
                    await _store_upload(storage_dir, log_file, prefix="log")
                )

            hardware_file = _as_upload(upload_fields["hardware_file"])
            if hardware_file:
                data.hardware_path = await _store_upload(
                    storage_dir, hardware_file, prefix="hardware"
                )

            telemetry_before = _as_upload(upload_fields["telemetry_before"])
            if telemetry_before:
                path, parsed = await _store_json_upload(
                    storage_dir, telemetry_before, prefix="baseline"
                )
                data.telemetry_before = parsed
                _maybe_set_model_name(data, parsed)

            telemetry_after = _as_upload(upload_fields["telemetry_after"])
            if telemetry_after:
                path, parsed = await _store_json_upload(
                    storage_dir, telemetry_after, prefix="optimized"
                )
                data.telemetry_after = parsed
                _maybe_set_model_name(data, parsed)

            usage_file = _as_upload(upload_fields["usage_event_file"])
            if usage_file:
                path, parsed = await _store_json_upload(
                    storage_dir, usage_file, prefix="usage"
                )
                data.usage_event = parsed
                data.usage_event_path = path
                _maybe_set_model_name(data, parsed)

            for key, label in (
                ("optimized_ir", "Optimized IR"),
                ("kernel_file", "Kernel Artifact"),
            ):
                upload = _as_upload(upload_fields[key])
                if upload:
                    artifact_path = await _store_upload(
                        storage_dir, upload, prefix="artifact"
                    )
                    data.kernel_artifacts.append(
                        KernelArtifact(
                            identifier=uuid.uuid4().hex,
                            name=_safe_filename(upload.filename),
                            path=artifact_path,
                            description=label,
                        )
                    )

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
                    "download_url": f"/artifacts/{artifact.identifier}",
                }
                for artifact in data.kernel_artifacts
            ]
        return JSONResponse({"artifacts": artifacts})

    async def download_artifact(request: Request) -> FileResponse:
        artifact_id = request.path_params["artifact_id"]
        async with lock:
            artifact = next(
                (
                    item
                    for item in data.kernel_artifacts
                    if item.identifier == artifact_id
                ),
                None,
            )
            if not artifact:
                raise HTTPException(status_code=404, detail="Artifact not found")
            path = artifact.path
            filename = artifact.name

        return FileResponse(
            path,
            media_type="application/octet-stream",
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
        Route("/artifacts/{artifact_id}", download_artifact, methods=["GET"]),
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


async def _store_upload(
    storage_dir: Path, upload: UploadFile, prefix: str
) -> Path:
    safe_name = _safe_filename(upload.filename)
    destination = storage_dir / f"{prefix}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    destination.write_bytes(content)
    return destination


async def _store_json_upload(
    storage_dir: Path, upload: UploadFile, prefix: str
) -> Tuple[Path, Dict[str, Any]]:
    safe_name = _safe_filename(upload.filename)
    destination = storage_dir / f"{prefix}_{safe_name}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = await upload.read()
    try:
        parsed = json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON payload for {safe_name}: {exc}",
        ) from exc
    destination.write_bytes(content)
    return destination, parsed


def _write_text_file(storage_dir: Path, filename: str, text: str) -> Path:
    path = storage_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


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
        for artifact in data.kernel_artifacts:
            if artifact.path.exists():
                arcname = f"artifacts/{artifact.name}"
                zf.write(artifact.path, arcname=arcname)

    buffer.seek(0)
    return buffer.read()


__all__ = [
    "DashboardData",
    "KernelArtifact",
    "build_layer_stats",
    "build_performance_summary",
    "build_sdk_pack_bytes",
    "create_app",
]
