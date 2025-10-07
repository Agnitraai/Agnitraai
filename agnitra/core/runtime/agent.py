"""Runtime optimization agent that ties together tuning, telemetry, and metering."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - PyTorch absent at runtime
    torch = None  # type: ignore[assignment]

from agnitra.core.metering import UsageEvent, UsageMeter
from agnitra.telemetry_collector import profile_model


def _clone_tensor(tensor: Any) -> Any:
    try:
        return tensor.clone().detach()
    except Exception:
        return tensor


def _count_tensor_tokens(tensor: Any) -> int:
    try:
        if torch is None:
            return 0
        if isinstance(tensor, torch.Tensor):
            return int(tensor.numel())
        return 0
    except Exception:
        return 0


def _infer_module_device(module: Any) -> Optional["torch.device"]:
    if torch is None:
        return None
    for accessor in ("parameters", "buffers"):
        if not hasattr(module, accessor):
            continue
        try:
            iterator = getattr(module, accessor)()  # type: ignore[call-arg]
        except Exception:
            continue
        for item in iterator:
            if isinstance(item, torch.Tensor):
                return item.device
    return None


def _optimize_model(model: Any, tensor: Any, enable_rl: bool) -> Any:
    from agnitra._sdk import optimizer as _optimizer  # Local import to avoid circular dependency

    return _optimizer.optimize_model(model, tensor, enable_rl=enable_rl)


@dataclass
class OptimizationSnapshot:
    """Point-in-time measurement captured before or after optimization."""

    latency_ms: float
    tokens_per_sec: float
    tokens_processed: int
    gpu_utilization: Optional[float]
    telemetry: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "latency_ms": self.latency_ms,
            "tokens_per_sec": self.tokens_per_sec,
            "tokens_processed": self.tokens_processed,
            "gpu_utilization": self.gpu_utilization,
            "metadata": dict(self.metadata),
        }
        payload["telemetry"] = self.telemetry
        return payload


@dataclass
class RuntimeOptimizationResult:
    """Full outcome from :class:`RuntimeOptimizationAgent.optimize`."""

    optimized_model: Any
    baseline: OptimizationSnapshot
    optimized: OptimizationSnapshot
    usage_event: Optional[UsageEvent]
    notes: Dict[str, Any] = field(default_factory=dict)


class RuntimeOptimizationAgent:
    """High-level orchestrator that profiles, optimizes, and meters usage."""

    def __init__(
        self,
        *,
        usage_meter: Optional[UsageMeter] = None,
        repeats: int = 10,
        warmup: int = 3,
        rate_per_gpu_hour: float = 2.5,
        success_margin_pct: float = 0.2,
    ) -> None:
        self.repeats = max(1, int(repeats))
        self.warmup = max(0, int(warmup))
        self._usage_meter = usage_meter or UsageMeter(
            rate_per_gpu_hour=rate_per_gpu_hour,
            margin_pct=success_margin_pct,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def usage_meter(self) -> UsageMeter:
        """Return the underlying :class:`UsageMeter` instance."""

        return self._usage_meter

    def optimize(
        self,
        model: Any,
        input_tensor: Any,
        *,
        project_id: str = "default",
        model_name: Optional[str] = None,
        enable_rl: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RuntimeOptimizationResult:
        """Optimize ``model`` while capturing baseline/optimized metrics."""

        torch_mod = self._require_torch()
        named_model = model_name or getattr(model, "__class__", type(model)).__name__

        sample = self._prepare_tensor(model, input_tensor, torch_mod)

        baseline_snapshot = self._capture_snapshot(
            model,
            sample,
            torch_mod,
            stage="baseline",
            extra_metadata=metadata,
        )

        optimized_model = _optimize_model(model, sample.clone(), enable_rl=enable_rl)

        optimized_snapshot = self._capture_snapshot(
            optimized_model,
            sample,
            torch_mod,
            stage="optimized",
            extra_metadata=metadata,
        )

        usage_event: Optional[UsageEvent] = None
        if self._usage_meter is not None:
            usage_event = self._usage_meter.record_optimization(
                project_id=project_id,
                model_name=named_model,
                baseline_snapshot=baseline_snapshot,
                optimized_snapshot=optimized_snapshot,
                tokens_processed=optimized_snapshot.tokens_processed,
                metadata={
                    **(metadata or {}),
                    "stage_notes": "baseline_vs_optimized",
                },
            )

        result = RuntimeOptimizationResult(
            optimized_model=optimized_model,
            baseline=baseline_snapshot,
            optimized=optimized_snapshot,
            usage_event=usage_event,
            notes={
                "project_id": project_id,
                "model_name": named_model,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _require_torch(self) -> "torch":
        if torch is None:  # pragma: no cover - handled when torch missing
            raise RuntimeError("PyTorch is required to run the runtime optimization agent.")
        return torch

    def _prepare_tensor(self, module: Any, tensor: Any, torch_mod: "torch") -> "torch.Tensor":
        if not isinstance(tensor, torch_mod.Tensor):
            raise TypeError("RuntimeOptimizationAgent expects a torch.Tensor as input_tensor.")
        device = _infer_module_device(module) or tensor.device
        prepared = _clone_tensor(tensor)
        if device and prepared.device != device:
            prepared = prepared.to(device)
        return prepared

    def _capture_snapshot(
        self,
        module: Any,
        tensor: "torch.Tensor",
        torch_mod: "torch",
        *,
        stage: str,
        extra_metadata: Optional[Dict[str, Any]],
    ) -> OptimizationSnapshot:
        timings = []
        module_was_training = getattr(module, "training", False)
        try:
            module.eval()
        except Exception:
            pass

        measured_tensor = _clone_tensor(tensor)
        try:
            if hasattr(measured_tensor, "device") and measured_tensor.device != tensor.device:
                measured_tensor = measured_tensor.to(tensor.device)
        except Exception:
            pass

        with torch_mod.inference_mode():
            for _ in range(self.warmup):
                try:
                    module(measured_tensor)
                    self._sync_device(torch_mod, getattr(measured_tensor, "device", None))
                except Exception:
                    break

            for _ in range(self.repeats):
                start = time.perf_counter()
                module(measured_tensor)
                self._sync_device(torch_mod, getattr(measured_tensor, "device", None))
                timings.append(time.perf_counter() - start)

        if module_was_training:
            try:
                module.train(True)
            except Exception:
                pass

        latency_ms = (sum(timings) / max(len(timings), 1)) * 1000.0
        tokens = _count_tensor_tokens(measured_tensor)
        tokens_per_sec = tokens / max(latency_ms / 1000.0, 1e-6) if tokens else 0.0

        telemetry = self._collect_telemetry(module, measured_tensor, torch_mod)
        gpu_util = self._extract_gpu_util(telemetry)

        snapshot = OptimizationSnapshot(
            latency_ms=latency_ms,
            tokens_per_sec=tokens_per_sec,
            tokens_processed=tokens,
            gpu_utilization=gpu_util,
            telemetry=telemetry,
            metadata={
                "stage": stage,
                **(extra_metadata or {}),
            },
        )
        return snapshot

    def _sync_device(self, torch_mod: "torch", device: Optional["torch.device"]) -> None:
        try:
            if device is not None and device.type == "cuda" and torch_mod.cuda.is_available():
                torch_mod.cuda.synchronize(device)
        except Exception:
            pass

    def _collect_telemetry(
        self,
        module: Any,
        tensor: "torch.Tensor",
        torch_mod: "torch",
    ) -> Dict[str, Any]:
        try:
            telemetry = profile_model(module, _clone_tensor(tensor))
            return telemetry
        except Exception:
            return {}

    def _extract_gpu_util(self, telemetry: Dict[str, Any]) -> Optional[float]:
        gpu_section = telemetry.get("gpu") if isinstance(telemetry, dict) else None
        if isinstance(gpu_section, dict):
            util = gpu_section.get("gpu_utilisation")
            if isinstance(util, (int, float)):
                return float(util)
        behavior = telemetry.get("behavior") if isinstance(telemetry, dict) else None
        if isinstance(behavior, dict):
            util = behavior.get("gpu_util_mean")
            if isinstance(util, (int, float)):
                return float(util)
        return None


__all__ = [
    "OptimizationSnapshot",
    "RuntimeOptimizationAgent",
    "RuntimeOptimizationResult",
]
