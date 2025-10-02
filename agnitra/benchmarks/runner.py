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

from agnitra._sdk.optimizer import optimize_model


@dataclass
class BenchResult:
    latency_ms: float
    memory_bytes: int
    repeats: int
    tokens_per_sec: float


def _infer_token_count(input_tensor: Any) -> int:
    """Best-effort token counter used for throughput reporting."""

    if input_tensor is None:
        return 0

    if torch is not None and isinstance(input_tensor, torch.Tensor):
        if input_tensor.ndim >= 2:
            batch = int(input_tensor.shape[0]) if input_tensor.shape else 1
            seq = int(input_tensor.shape[1]) if input_tensor.ndim > 1 else 1
            return max(batch * seq, 0)
        return max(int(input_tensor.numel()), 0)

    # Allow callers to pre-compute tokens via attribute for custom objects
    token_attr = getattr(input_tensor, "agnitra_token_count", None)
    if isinstance(token_attr, int):
        return max(token_attr, 0)

    return 0


def _measure(
    model: Any,
    input_tensor: Any,
    repeats: int = 10,
    warmup: int = 2,
    token_count: Optional[int] = None,
) -> BenchResult:
    tokens = token_count if token_count is not None else _infer_token_count(input_tensor)

    if torch is None:
        for _ in range(max(0, warmup)):
            _ = model(input_tensor)
        total_iters = max(1, repeats)
        t0 = time.perf_counter()
        for _ in range(total_iters):
            _ = model(input_tensor)
        t1 = time.perf_counter()
        total_time = max(t1 - t0, 1e-9)
        latency_ms = (t1 - t0) * 1000.0 / total_iters
        tokens_per_sec = (float(tokens) * total_iters / total_time) if tokens else 0.0
        return BenchResult(
            latency_ms=latency_ms,
            memory_bytes=0,
            repeats=repeats,
            tokens_per_sec=tokens_per_sec,
        )

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

    total_iters = max(1, repeats)
    t0 = time.perf_counter()
    for _ in range(total_iters):
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
    latency_ms = (t1 - t0) * 1000.0 / max(1, repeats)
    total_time = max(t1 - t0, 1e-9)
    tokens_per_sec = (float(tokens) * total_iters / total_time) if tokens else 0.0

    return BenchResult(
        latency_ms=latency_ms,
        memory_bytes=mem,
        repeats=repeats,
        tokens_per_sec=tokens_per_sec,
    )


def run_benchmark(
    model: Any,
    input_tensor: Any,
    out_dir: str | Path,
    repeats: int = 10,
    warmup: int = 2,
    enable_rl: bool = False,
    client: Optional[Any] = None,
    token_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Run baseline and optimized measurements and save JSON outputs.

    Returns an in-memory dict with the summary.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    before = _measure(
        model,
        input_tensor,
        repeats=repeats,
        warmup=warmup,
        token_count=token_count,
    )

    opt_model = optimize_model(model, input_tensor, client=client, enable_rl=enable_rl)
    after = _measure(
        opt_model,
        input_tensor,
        repeats=repeats,
        warmup=warmup,
        token_count=token_count,
    )

    before_json = out_path / "before.json"
    after_json = out_path / "after.json"
    summary_json = out_path / "summary.json"
    summary_diff_json = out_path / "summary_diff.json"

    with before_json.open("w", encoding="utf-8") as fh:
        json.dump(asdict(before), fh, indent=2)
    with after_json.open("w", encoding="utf-8") as fh:
        json.dump(asdict(after), fh, indent=2)

    speedup = (before.latency_ms / after.latency_ms) if after.latency_ms > 0 else 1.0
    mem_saving = (before.memory_bytes - after.memory_bytes)
    token_delta = after.tokens_per_sec - before.tokens_per_sec
    memory_before_gb = before.memory_bytes / (1024**3)
    memory_after_gb = after.memory_bytes / (1024**3)
    summary = {
        "speedup": float(speedup),
        "latency_before_ms": float(before.latency_ms),
        "latency_after_ms": float(after.latency_ms),
        "memory_before_bytes": int(before.memory_bytes),
        "memory_after_bytes": int(after.memory_bytes),
        "memory_saving_bytes": int(mem_saving),
        "repeats": int(repeats),
        "warmup": int(warmup),
        "tokens_before_per_sec": float(before.tokens_per_sec),
        "tokens_after_per_sec": float(after.tokens_per_sec),
        "tokens_per_sec_delta": float(token_delta),
        "memory_before_gb": float(memory_before_gb),
        "memory_after_gb": float(memory_after_gb),
        "results": {
            "baseline": asdict(before),
            "optimized": asdict(after),
        },
    }
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with summary_diff_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "speedup": summary["speedup"],
                "latency_delta_ms": summary["latency_after_ms"] - summary["latency_before_ms"],
                "memory_delta_bytes": summary["memory_after_bytes"] - summary["memory_before_bytes"],
                "tokens_delta_per_sec": summary["tokens_per_sec_delta"],
            },
            fh,
            indent=2,
        )

    return summary


__all__ = ["run_benchmark", "BenchResult"]
