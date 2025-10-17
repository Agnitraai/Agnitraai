"""Cost delta calculations for per-inference uplift billing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


@dataclass
class BenchmarkSample:
    """Before/after benchmark artefacts used for uplift estimation."""

    baseline_latency_ms: float
    optimized_latency_ms: float
    baseline_tokens_per_sec: float
    optimized_tokens_per_sec: float
    tokens_processed: Optional[int] = None
    currency: str = "USD"


@dataclass
class CostDeltaResult:
    """Output structure returned by :func:`compute_cost_delta`."""

    cost_before: float
    cost_after: float
    cost_saving: float
    latency_delta_ms: float
    tokens_per_dollar_uplift: float
    tokens_per_sec_uplift_pct: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "cost_saving": self.cost_saving,
            "latency_delta_ms": self.latency_delta_ms,
            "tokens_per_dollar_uplift": self.tokens_per_dollar_uplift,
            "tokens_per_sec_uplift_pct": self.tokens_per_sec_uplift_pct,
        }


def summarise_benchmark(payload: Mapping[str, Any]) -> BenchmarkSample:
    """Build a :class:`BenchmarkSample` from ``benchmark_runner`` output."""

    baseline = payload.get("baseline", {}) if isinstance(payload, Mapping) else {}
    optimized = payload.get("optimized", {}) if isinstance(payload, Mapping) else {}

    return BenchmarkSample(
        baseline_latency_ms=_safe_float(baseline.get("latency_ms")),
        optimized_latency_ms=_safe_float(optimized.get("latency_ms")),
        baseline_tokens_per_sec=_safe_float(baseline.get("tokens_per_sec")),
        optimized_tokens_per_sec=_safe_float(optimized.get("tokens_per_sec")),
        tokens_processed=int(_safe_float(payload.get("tokens_processed"))) or None,
        currency=str(payload.get("currency", "USD")),
    )


def compute_cost_delta(sample: BenchmarkSample, *, cost_per_second_gpu: float) -> CostDeltaResult:
    """Estimate cost deltas for uplift-based billing models."""

    latency_delta = max(sample.baseline_latency_ms - sample.optimized_latency_ms, 0.0)
    baseline_seconds = sample.baseline_latency_ms / 1000.0
    optimized_seconds = sample.optimized_latency_ms / 1000.0

    cost_before = baseline_seconds * cost_per_second_gpu
    cost_after = optimized_seconds * cost_per_second_gpu
    cost_saving = max(cost_before - cost_after, 0.0)

    baseline_tps = max(sample.baseline_tokens_per_sec, 1e-6)
    optimized_tps = max(sample.optimized_tokens_per_sec, 1e-6)
    tokens_per_sec_uplift_pct = ((optimized_tps - baseline_tps) / baseline_tps) * 100.0

    tokens_per_dollar_before = baseline_tps / max(cost_before, 1e-6)
    tokens_per_dollar_after = optimized_tps / max(cost_after, 1e-6)
    tokens_per_dollar_uplift = max(tokens_per_dollar_after - tokens_per_dollar_before, 0.0)

    return CostDeltaResult(
        cost_before=cost_before,
        cost_after=cost_after,
        cost_saving=cost_saving,
        latency_delta_ms=latency_delta,
        tokens_per_dollar_uplift=tokens_per_dollar_uplift,
        tokens_per_sec_uplift_pct=tokens_per_sec_uplift_pct,
    )
