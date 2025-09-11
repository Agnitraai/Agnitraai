"""Benchmark runner comparing baseline vs optimized performance.

This utility measures latency and (optionally) CUDA memory for a PyTorch
model before and after optimization via the Agnitra SDK. It writes
``before.json``, ``after.json`` and ``summary.json`` into the specified
output directory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None

from agnitra.sdk.optimizer import optimize_model


@dataclass
class BenchResult:
    latency_ms: float
    memory_bytes: int
    repeats: int


def _measure(model: Any, input_tensor: Any, repeats: int = 10, warmup: int = 2) -> BenchResult:
    if torch is None:
        t0 = time.perf_counter()
        for _ in range(repeats + warmup):
            _ = model(input_tensor)
        t1 = time.perf_counter()
        # no GPU memory metrics on CPU-only
        return BenchResult(latency_ms=(t1 - t0) * 1000.0 / max(1, (repeats + warmup)), memory_bytes=0, repeats=repeats)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device) if hasattr(model, "to") else model
    input_tensor = input_tensor.to(device) if hasattr(input_tensor, "to") else input_tensor

    if device == "cuda":  # pragma: no branch
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(max(0, warmup)):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    mem = 0
    if device == "cuda":
        try:
            mem = int(torch.cuda.max_memory_allocated())
        except Exception:
            mem = 0

    return BenchResult(latency_ms=(t1 - t0) * 1000.0 / max(1, repeats), memory_bytes=mem, repeats=repeats)


def run_benchmark(
    model: Any,
    input_tensor: Any,
    out_dir: str | Path,
    repeats: int = 10,
    warmup: int = 2,
    enable_rl: bool = False,
    client: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run baseline and optimized measurements and save JSON outputs.

    Returns an in-memory dict with the summary.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    before = _measure(model, input_tensor, repeats=repeats, warmup=warmup)

    opt_model = optimize_model(model, input_tensor, client=client, enable_rl=enable_rl)
    after = _measure(opt_model, input_tensor, repeats=repeats, warmup=warmup)

    before_json = out_path / "before.json"
    after_json = out_path / "after.json"
    summary_json = out_path / "summary.json"

    with before_json.open("w", encoding="utf-8") as fh:
        json.dump(asdict(before), fh, indent=2)
    with after_json.open("w", encoding="utf-8") as fh:
        json.dump(asdict(after), fh, indent=2)

    speedup = (before.latency_ms / after.latency_ms) if after.latency_ms > 0 else 1.0
    mem_saving = (before.memory_bytes - after.memory_bytes)
    summary = {
        "speedup": float(speedup),
        "latency_before_ms": float(before.latency_ms),
        "latency_after_ms": float(after.latency_ms),
        "memory_before_bytes": int(before.memory_bytes),
        "memory_after_bytes": int(after.memory_bytes),
        "memory_saving_bytes": int(mem_saving),
        "repeats": int(repeats),
        "warmup": int(warmup),
    }
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    return summary


__all__ = ["run_benchmark", "BenchResult"]

