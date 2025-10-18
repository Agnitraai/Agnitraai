"""Structured metrics logging for the hosted optimization API."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _default_log_path() -> Path:
    candidate = os.environ.get("AGNITRA_METRICS_LOG")
    if candidate:
        return Path(candidate).expanduser()
    return Path("agnitra_metrics.jsonl")


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


@dataclass
class MetricsLogger:
    """Append-only JSONL logger for optimization performance snapshots."""

    path: Path = field(default_factory=_default_log_path)
    ensure_dir: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _prepare_directory(self) -> None:
        if not self.ensure_dir:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Best-effort; failures bubble up when writing.
            pass

    def log(
        self,
        *,
        project_id: str,
        model_name: Optional[str],
        target: str,
        telemetry_summary: Mapping[str, Any],
        patch_metrics: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Append a single optimization metrics entry."""

        entry: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "model_name": model_name,
            "target": target,
            "total_latency_ms": _safe_float(telemetry_summary.get("total_latency_ms")),
            "event_count": _safe_int(telemetry_summary.get("event_count")),
            "baseline_latency_ms": _safe_float(patch_metrics.get("baseline_latency_ms")),
            "expected_latency_ms": _safe_float(patch_metrics.get("expected_latency_ms")),
            "expected_speedup_pct": _safe_float(patch_metrics.get("expected_speedup_pct")),
            "metadata": dict(metadata or {}),
        }

        baseline_latency = entry["baseline_latency_ms"]
        expected_latency = entry["expected_latency_ms"]
        if baseline_latency > 0:
            entry["expected_tokens_per_sec"] = 1000.0 / expected_latency if expected_latency > 0 else None
            entry["baseline_tokens_per_sec"] = 1000.0 / baseline_latency
            entry["latency_delta_ms"] = baseline_latency - expected_latency
        else:
            entry["expected_tokens_per_sec"] = None
            entry["baseline_tokens_per_sec"] = None
            entry["latency_delta_ms"] = None

        self._write_entry(entry)
        return entry

    def _write_entry(self, entry: Mapping[str, Any]) -> None:
        self._prepare_directory()
        encoded = json.dumps(entry, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.write("\n")

