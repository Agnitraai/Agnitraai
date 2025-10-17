"""Embedded runtime helpers for OEM partnerships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass
class EmbeddedTelemetry:
    tokens_per_sec: float
    latency_ms: float
    memory_bytes: Optional[int] = None
    extra: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "tokens_per_sec": self.tokens_per_sec,
            "latency_ms": self.latency_ms,
            "memory_bytes": self.memory_bytes,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


def suggest_kernel_config(target: str, telemetry: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a lightweight kernel config for embedded devices."""

    base = {
        "target": target,
        "block_size": 128 if target.lower().startswith("gpu") else 64,
        "num_warps": 2 if "cuda" in target.lower() else 1,
    }
    if "latency_ms" in telemetry:
        base["latency_budget_ms"] = telemetry["latency_ms"]
    if "memory_bytes" in telemetry:
        base["memory_budget_bytes"] = telemetry["memory_bytes"]
    return base
