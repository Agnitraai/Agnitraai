"""Usage metering and billing utilities for the pay-per-optimization flow."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


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


@dataclass
class UsageEvent:
    """Structured usage entry produced after an optimization run."""

    project_id: str
    model_name: str
    tokens_processed: int
    baseline_latency_ms: float
    optimized_latency_ms: float
    baseline_tokens_per_sec: float
    optimized_tokens_per_sec: float
    gpu_util_before: Optional[float]
    gpu_util_after: Optional[float]
    gpu_hours_before: float
    gpu_hours_after: float
    gpu_hours_saved: float
    performance_uplift_pct: float
    cost_before: float
    cost_after: float
    cost_savings: float
    usage_charge: float
    success_fee: float
    total_billable: float
    currency: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of the usage event."""

        return {
            "project_id": self.project_id,
            "model_name": self.model_name,
            "tokens_processed": self.tokens_processed,
            "baseline_latency_ms": self.baseline_latency_ms,
            "optimized_latency_ms": self.optimized_latency_ms,
            "baseline_tokens_per_sec": self.baseline_tokens_per_sec,
            "optimized_tokens_per_sec": self.optimized_tokens_per_sec,
            "gpu_util_before": self.gpu_util_before,
            "gpu_util_after": self.gpu_util_after,
            "gpu_hours_before": self.gpu_hours_before,
            "gpu_hours_after": self.gpu_hours_after,
            "gpu_hours_saved": self.gpu_hours_saved,
            "performance_uplift_pct": self.performance_uplift_pct,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "cost_savings": self.cost_savings,
            "usage_charge": self.usage_charge,
            "success_fee": self.success_fee,
            "total_billable": self.total_billable,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }


class UsageMeter:
    """Usage meter that converts optimization snapshots into billable records."""

    def __init__(
        self,
        *,
        rate_per_gpu_hour: float = 2.5,
        margin_pct: float = 0.2,
        currency: str = "USD",
    ) -> None:
        self.rate_per_gpu_hour = _safe_float(rate_per_gpu_hour, 2.5)
        self.margin_pct = max(0.0, _safe_float(margin_pct, 0.2))
        self.currency = currency
        self._events: List[UsageEvent] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_optimization(
        self,
        *,
        project_id: str,
        model_name: str,
        baseline_snapshot: Any,
        optimized_snapshot: Any,
        tokens_processed: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UsageEvent:
        """Create and store a :class:`UsageEvent` from snapshot data."""

        baseline_latency_ms = _safe_float(getattr(baseline_snapshot, "latency_ms", None))
        optimized_latency_ms = _safe_float(getattr(optimized_snapshot, "latency_ms", None))
        baseline_tps = max(_safe_float(getattr(baseline_snapshot, "tokens_per_sec", None)), 1e-6)
        optimized_tps = max(_safe_float(getattr(optimized_snapshot, "tokens_per_sec", None)), 1e-6)
        baseline_util = getattr(baseline_snapshot, "gpu_utilization", None)
        optimized_util = getattr(optimized_snapshot, "gpu_utilization", None)

        inferred_tokens = tokens_processed
        if inferred_tokens is None:
            inferred_tokens = getattr(optimized_snapshot, "tokens_processed", None)
        if inferred_tokens is None:
            inferred_tokens = getattr(baseline_snapshot, "tokens_processed", None)
        tokens = max(_safe_int(inferred_tokens, 0), 0)

        if tokens <= 0:
            baseline_seconds = baseline_latency_ms / 1000.0
            optimized_seconds = optimized_latency_ms / 1000.0
        else:
            baseline_seconds = tokens / baseline_tps
            optimized_seconds = tokens / optimized_tps

        baseline_gpu_hours = max(baseline_seconds, 0.0) / 3600.0
        optimized_gpu_hours = max(optimized_seconds, 0.0) / 3600.0
        gpu_hours_saved = max(baseline_gpu_hours - optimized_gpu_hours, 0.0)

        perf_uplift_pct = 0.0
        try:
            perf_uplift_pct = ((optimized_tps - baseline_tps) / baseline_tps) * 100.0
        except ZeroDivisionError:
            perf_uplift_pct = 0.0

        cost_before = baseline_gpu_hours * self.rate_per_gpu_hour
        cost_after = optimized_gpu_hours * self.rate_per_gpu_hour
        cost_savings = max(cost_before - cost_after, 0.0)

        usage_charge = cost_after
        success_fee = cost_savings * self.margin_pct
        total_billable = usage_charge + success_fee

        event = UsageEvent(
            project_id=project_id,
            model_name=model_name,
            tokens_processed=tokens,
            baseline_latency_ms=baseline_latency_ms,
            optimized_latency_ms=optimized_latency_ms,
            baseline_tokens_per_sec=baseline_tps,
            optimized_tokens_per_sec=optimized_tps,
            gpu_util_before=baseline_util,
            gpu_util_after=optimized_util,
            gpu_hours_before=baseline_gpu_hours,
            gpu_hours_after=optimized_gpu_hours,
            gpu_hours_saved=gpu_hours_saved,
            performance_uplift_pct=perf_uplift_pct,
            cost_before=cost_before,
            cost_after=cost_after,
            cost_savings=cost_savings,
            usage_charge=usage_charge,
            success_fee=success_fee,
            total_billable=total_billable,
            currency=self.currency,
            metadata=dict(metadata or {}),
        )
        self._events.append(event)
        return event

    def all_events(self) -> List[UsageEvent]:
        """Return a snapshot of all logged usage events."""

        return list(self._events)

    def extend(self, events: Iterable[UsageEvent]) -> None:
        """Append multiple usage events to the meter."""

        for event in events:
            if isinstance(event, UsageEvent):
                self._events.append(event)

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Return all usage events as dictionaries."""

        return [event.to_dict() for event in self._events]


__all__ = ["UsageMeter", "UsageEvent"]
