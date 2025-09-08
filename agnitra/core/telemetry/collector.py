"""Telemetry event collector.

Utilities to gather metrics from profiling events. When running in a CPU-only
environment, profiling events may not expose CUDA-specific attributes. This
module gracefully handles their absence by defaulting them to ``0.0`` for time
and ``0`` for memory usage."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def collect_events(events: Iterable[Any]) -> List[Dict[str, float | int | str]]:
    """Collect telemetry information from an iterable of events.

    Parameters
    ----------
    events:
        Iterable of profiler events. Each event may define the following
        attributes depending on the available hardware during profiling:

        * ``key``: Identifier for the event.
        * ``cpu_time_total``: Total CPU time spent in the event.
        * ``cuda_time_total``: Total CUDA execution time for the event.
        * ``self_cuda_memory_usage``: Amount of CUDA memory used by the event.

        Any of these attributes may be absent, in which case sensible defaults
        (``""`` for ``key`` and ``0``/``0.0`` for numeric fields) are used.

    Returns
    -------
    list of dict
        A list of dictionaries containing the collected metrics for each
        event. Each dictionary has the keys ``key``, ``cpu_time_total``,
        ``cuda_time_total`` and ``self_cuda_memory_usage``.
    """

    collected: List[Dict[str, float | int | str]] = []
    for evt in events:
        # Guard against missing attributes when running without GPUs or when
        # profiling produced limited data.
        key = getattr(evt, "key", "")
        cpu_time_total = getattr(evt, "cpu_time_total", 0.0)
        cuda_time_total = getattr(evt, "cuda_time_total", 0.0)
        self_cuda_memory_usage = getattr(evt, "self_cuda_memory_usage", 0)

        collected.append(
            {
                "key": str(key),
                "cpu_time_total": float(cpu_time_total),
                "cuda_time_total": float(cuda_time_total),
                "self_cuda_memory_usage": int(self_cuda_memory_usage),
            }
        )

    return collected
