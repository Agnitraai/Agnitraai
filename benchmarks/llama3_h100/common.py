"""Shared helpers for runners.

Anything timing-, memory-, or environment-related lives here so all
runners produce comparable measurements.
"""
from __future__ import annotations

import datetime as _dt
import statistics
import time
from contextlib import contextmanager
from typing import Iterator, List, Tuple

from .schema import Latency


def utc_timestamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def gpu_info() -> Tuple[str, str, str]:
    """Return (gpu_name, cuda_version, torch_version).

    Falls back to "unknown" strings rather than raising — the benchmark
    should still produce a JSON file even when run on a host without a
    properly detected GPU, so failures are visible in RESULTS.md.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - defensive
        return ("no-cuda", "unknown", "unknown")

    torch_version = getattr(torch, "__version__", "unknown")
    if not torch.cuda.is_available():
        return ("no-cuda", "unknown", torch_version)

    name = torch.cuda.get_device_name(0)
    cuda_version = getattr(torch.version, "cuda", "unknown") or "unknown"
    return (name, cuda_version, torch_version)


@contextmanager
def cuda_memory_tracker() -> Iterator[List[float]]:
    """Resets and captures peak GPU memory in GB for the with-block.

    Yields a list with a single float that is populated on exit. Using
    a list keeps the API simple while letting the caller read the value
    after the context closes.
    """
    holder: List[float] = [0.0]
    try:
        import torch
    except Exception:
        yield holder
        return

    if not torch.cuda.is_available():
        yield holder
        return

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    yield holder
    torch.cuda.synchronize()
    holder[0] = torch.cuda.max_memory_allocated() / (1024 ** 3)


def cuda_sync() -> None:
    """Synchronize the default CUDA stream if torch+CUDA are available."""
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_block_ms(fn, iterations: int) -> List[float]:
    """Run ``fn`` ``iterations`` times and return per-call wall time in ms.

    ``fn`` is responsible for any internal CUDA syncing it needs; this
    helper only syncs around each call boundary.
    """
    samples: List[float] = []
    for _ in range(iterations):
        cuda_sync()
        t0 = time.perf_counter()
        fn()
        cuda_sync()
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000.0)
    return samples


def latency_from_samples(samples: List[float]) -> Latency:
    if not samples:
        return Latency(p50_ms=0.0, p99_ms=0.0, mean_ms=0.0, samples_ms=[])
    sorted_samples = sorted(samples)
    p50 = statistics.median(sorted_samples)
    # For 10 samples, p99 collapses to max — that's fine and documented.
    idx99 = max(0, min(len(sorted_samples) - 1, int(round(0.99 * (len(sorted_samples) - 1)))))
    p99 = sorted_samples[idx99]
    mean = statistics.fmean(sorted_samples)
    return Latency(p50_ms=p50, p99_ms=p99, mean_ms=mean, samples_ms=samples)
