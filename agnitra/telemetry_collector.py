"""Advanced telemetry collector using ``torch.profiler``.

This module profiles a PyTorch model and records per-layer metrics such as
CUDA time, tensor shapes and memory usage. Additionally, if NVIDIA's NVML is
available, GPU utilisation and power draw are logged. Results are optionally
written to a JSON file for further analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch.profiler import ProfilerActivity, profile
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None
    ProfilerActivity = profile = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from pynvml import (
        NVMLError,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
    )

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NVML_AVAILABLE = False


@dataclass
class EventTelemetry:
    """Telemetry for a single profiler event."""

    name: str
    cuda_time_total: float
    self_cuda_memory_usage: int
    input_shapes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cuda_time_total": self.cuda_time_total,
            "self_cuda_memory_usage": self.self_cuda_memory_usage,
            "input_shapes": self.input_shapes,
        }


@dataclass
class GpuTelemetry:
    """High level GPU metrics."""

    gpu_utilisation: int | None = None
    memory_utilisation: int | None = None
    power_watts: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_utilisation": self.gpu_utilisation,
            "memory_utilisation": self.memory_utilisation,
            "power_watts": self.power_watts,
        }


def _capture_gpu_metrics() -> GpuTelemetry:
    """Best effort capture of GPU metrics using NVML."""

    if not _NVML_AVAILABLE:
        return GpuTelemetry()

    try:  # pragma: no cover - requires GPU
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        util = nvmlDeviceGetUtilizationRates(handle)
        power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        return GpuTelemetry(util.gpu, util.memory, power)
    except NVMLError:
        return GpuTelemetry()
    finally:  # pragma: no cover - requires GPU
        try:
            nvmlShutdown()
        except Exception:
            pass


def profile_model(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    input_tensor: "torch.Tensor",  # type: ignore[name-defined]
    json_path: str | None = None,
) -> Dict[str, Any]:
    """Profile ``model`` using ``input_tensor``.

    When PyTorch is unavailable, an empty telemetry payload is returned and,
    if ``json_path`` is provided, written to disk. This allows callers to
    gracefully proceed in CPU-only environments.
    """

    if torch is None or profile is None or ProfilerActivity is None:
        logging.warning("PyTorch not available; returning empty telemetry")
        payload = {"events": [], "gpu": {}}
        if json_path is not None:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        return payload

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():  # pragma: no branch
        activities.append(ProfilerActivity.CUDA)
        model = model.to("cuda")
        input_tensor = input_tensor.to("cuda")

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        model(input_tensor)

    events: List[EventTelemetry] = []
    for evt in prof.key_averages():
        events.append(
            EventTelemetry(
                name=str(evt.key),
                cuda_time_total=float(getattr(evt, "cuda_time_total", 0.0)),
                self_cuda_memory_usage=int(
                    getattr(evt, "self_cuda_memory_usage", 0)
                ),
                input_shapes=[str(s) for s in getattr(evt, "input_shapes", [])],
            )
        )

    gpu = _capture_gpu_metrics()

    payload = {
        "events": [e.to_dict() for e in events],
        "gpu": gpu.to_dict(),
    }

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    return payload


__all__ = ["profile_model", "EventTelemetry", "GpuTelemetry"]
