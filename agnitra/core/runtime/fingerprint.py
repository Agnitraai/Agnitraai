"""Workload fingerprinting utilities for optimization runs."""

from __future__ import annotations

import hashlib
import json
import platform
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

try:  # pragma: no cover - torch optional
    import torch
except Exception:  # pragma: no cover - torch absent
    torch = None  # type: ignore


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _tensor_signature(tensor: Any) -> Dict[str, Any]:
    if torch is None:
        return {}
    if not isinstance(tensor, torch.Tensor):
        return {}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": bool(tensor.requires_grad),
    }


def _model_param_count(model: Any) -> int:
    if torch is None or not hasattr(model, "parameters"):
        return 0
    try:
        return sum(_safe_int(p.numel()) for p in model.parameters())  # type: ignore[attr-defined]
    except Exception:
        return 0


def _model_buffer_count(model: Any) -> int:
    if torch is None or not hasattr(model, "buffers"):
        return 0
    try:
        return sum(_safe_int(b.numel()) for b in model.buffers())  # type: ignore[attr-defined]
    except Exception:
        return 0


def _gpu_fingerprint() -> Dict[str, Any]:
    if torch is None or not torch.cuda.is_available():  # type: ignore[return-value]
        return {
            "vendor": "cpu",
            "model": platform.machine(),
            "driver": None,
            "count": 0,
        }
    try:
        idx = torch.cuda.current_device()
    except Exception:
        idx = 0
    try:
        name = torch.cuda.get_device_name(idx)
    except Exception:
        name = "unknown"
    try:
        capability = torch.cuda.get_device_capability(idx)
    except Exception:
        capability = None
    try:
        total_memory = torch.cuda.get_device_properties(idx).total_memory
    except Exception:
        total_memory = None
    try:
        driver_version = torch.version.cuda  # type: ignore[attr-defined]
    except Exception:
        driver_version = None
    return {
        "vendor": "nvidia",  # default assumption for CUDA
        "model": name,
        "capability": capability,
        "total_memory_bytes": total_memory,
        "driver": driver_version,
        "count": torch.cuda.device_count(),
    }


def _framework_info() -> Dict[str, Any]:
    if torch is None:
        return {"framework": "none"}
    torch_version = getattr(torch, "__version__", None)
    try:
        backend = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore[return-value]
    except Exception:
        backend = "unknown"
    return {"framework": "torch", "version": torch_version, "backend": backend}


def _hostname() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return "unknown-host"


@dataclass
class WorkloadFingerprint:
    """Structured description of the workload being optimized."""

    model_name: str
    param_count: int
    buffer_count: int
    framework: Mapping[str, Any]
    gpu: Mapping[str, Any]
    input_signature: Mapping[str, Any]
    extra_metadata: Mapping[str, Any]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "model_name": self.model_name,
            "param_count": self.param_count,
            "buffer_count": self.buffer_count,
            "framework": dict(self.framework),
            "gpu": dict(self.gpu),
            "input_signature": dict(self.input_signature),
            "extra_metadata": dict(self.extra_metadata),
            "created_at": self.created_at,
            "host": _hostname(),
        }
        return payload


def fingerprint_workload(
    model: Any,
    sample_inputs: Sequence[Any] | Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> WorkloadFingerprint:
    """Build a fingerprint combining model, framework, and device attributes."""

    model_cls = getattr(model, "__class__", None)
    model_name = getattr(model, "__name__", None)
    if model_cls is not None:
        model_name = getattr(model_cls, "__name__", model_name)
    model_name = model_name or "AnonymousModel"

    if isinstance(sample_inputs, (list, tuple)):
        tensor_sig = _tensor_signature(sample_inputs[0]) if sample_inputs else {}
    else:
        tensor_sig = _tensor_signature(sample_inputs)

    fingerprint = WorkloadFingerprint(
        model_name=model_name,
        param_count=_model_param_count(model),
        buffer_count=_model_buffer_count(model),
        framework=_framework_info(),
        gpu=_gpu_fingerprint(),
        input_signature=tensor_sig,
        extra_metadata=dict(metadata or {}),
        created_at=time.time(),
    )
    return fingerprint


def fingerprint_signature(payload: Mapping[str, Any]) -> str:
    """Generate a stable signature for the fingerprint mapping."""

    try:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError:
        encoded = json.dumps(json.loads(json.dumps(payload, default=str)), sort_keys=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return digest


__all__ = ["fingerprint_workload", "fingerprint_signature", "WorkloadFingerprint"]

