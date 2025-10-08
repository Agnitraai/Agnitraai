"""Runtime optimization agent that ties together tuning, telemetry, and metering."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

try:  # pragma: no cover - optional dependency guard
    import torch
except Exception:  # pragma: no cover - PyTorch absent at runtime
    torch = None  # type: ignore[assignment]

from agnitra.core.metering import UsageEvent, UsageMeter
from agnitra.core.runtime.cache import CachedProfile
from agnitra.core.runtime.control_plane import OptimizationPolicy
from agnitra.core.runtime.telemetry_client import TelemetryClient
from agnitra.telemetry_collector import profile_model

LOGGER = logging.getLogger(__name__)


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


def _optimize_model(
    model: Any,
    tensor: Any,
    enable_rl: bool,
    *,
    preset: Optional[Mapping[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> Any:
    from agnitra._sdk import optimizer as _optimizer  # Local import to avoid circular dependency

    return _optimizer.optimize_model(
        model,
        tensor,
        enable_rl=enable_rl,
        preset=preset,
        context=context,
        policy=policy,
    )


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
        policy: Optional[OptimizationPolicy] = None,
        cached_profile: Optional[CachedProfile] = None,
        telemetry_client: Optional[TelemetryClient] = None,
        fingerprint: Optional[Mapping[str, Any]] = None,
        fingerprint_signature: Optional[str] = None,
        cache_signature: Optional[str] = None,
    ) -> RuntimeOptimizationResult:
        """Optimize ``model`` while capturing baseline/optimized metrics."""

        torch_mod = self._require_torch()
        named_model = model_name or getattr(model, "__class__", type(model)).__name__

        sample = self._prepare_tensor(model, input_tensor, torch_mod)
        calibration_warmup = max(0, int(policy.calibration_warmup if policy else self.warmup))
        calibration_iterations = max(1, int(policy.calibration_iterations if policy else self.repeats))

        meta_base: Dict[str, Any] = dict(metadata or {})
        meta_base.setdefault("project_id", project_id)
        meta_base["policy_id"] = policy.policy_id if policy else None
        meta_base["cache_hit"] = bool(cached_profile)
        meta_base["cache_signature"] = cache_signature

        baseline_snapshot = self._capture_snapshot(
            model,
            sample,
            torch_mod,
            stage="baseline",
            repeats=calibration_iterations,
            warmup=calibration_warmup,
            extra_metadata=meta_base,
        )

        optimization_context: Dict[str, Any] = {
            "cache_hit": bool(cached_profile),
            "policy_id": policy.policy_id if policy else None,
            "fingerprint_signature": fingerprint_signature,
        }

        preset_override: Optional[Mapping[str, Any]] = None
        if cached_profile:
            cached_payload = dict(cached_profile.payload)
            cached_preset = cached_payload.get("applied_preset")
            if isinstance(cached_preset, dict):
                preset_override = cached_preset
                optimization_context["preset_source"] = "cache"
            optimization_context["cached_created_at"] = cached_profile.created_at
            optimization_context["cached_ttl_seconds"] = cached_profile.ttl_seconds
        elif policy and policy.default_preset:
            preset_override = dict(policy.default_preset)
            optimization_context["preset_source"] = "policy"
        else:
            optimization_context["preset_source"] = "discovery"

        effective_enable_rl = bool(enable_rl and (policy.enable_rl if policy else True))
        optimization_context["rl_enabled"] = effective_enable_rl
        optimization_context["auto_retrain_requested"] = bool(policy.auto_retrain if policy else False)
        policy_payload = policy.to_dict() if policy else None

        optimized_model = _optimize_model(
            model,
            sample.clone(),
            enable_rl=effective_enable_rl,
            preset=preset_override,
            context=optimization_context,
            policy=policy_payload,
        )

        optimized_snapshot = self._capture_snapshot(
            optimized_model,
            sample,
            torch_mod,
            stage="optimized",
            repeats=calibration_iterations,
            warmup=calibration_warmup,
            extra_metadata=meta_base,
        )

        usage_event: Optional[UsageEvent] = None
        if self._usage_meter is not None:
            meter_metadata = {
                **meta_base,
                "stage_notes": "baseline_vs_optimized",
                "fingerprint_signature": fingerprint_signature,
            }
            usage_event = self._usage_meter.record_optimization(
                project_id=project_id,
                model_name=named_model,
                baseline_snapshot=baseline_snapshot,
                optimized_snapshot=optimized_snapshot,
                tokens_processed=optimized_snapshot.tokens_processed,
                metadata=meter_metadata,
            )

        run_id = uuid.uuid4().hex
        auto_retrain_needed = bool(policy and policy.auto_retrain)
        optimization_context["auto_retrain_scheduled"] = auto_retrain_needed

        usage_payload = self._build_usage_payload(
            project_id=project_id,
            run_id=run_id,
            named_model=named_model,
            baseline_snapshot=baseline_snapshot,
            optimized_snapshot=optimized_snapshot,
            usage_event=usage_event,
            policy=policy,
            fingerprint=fingerprint,
            fingerprint_signature=fingerprint_signature or cache_signature,
            optimization_context=optimization_context,
        )

        if telemetry_client is not None:
            try:
                telemetry_client.emit(usage_payload)
            except Exception:  # pragma: no cover - telemetry best effort
                LOGGER.exception("Failed to emit telemetry usage payload")

        if auto_retrain_needed:
            self._schedule_auto_retrain(model, sample.clone(), torch_mod, policy_payload or {})

        result = RuntimeOptimizationResult(
            optimized_model=optimized_model,
            baseline=baseline_snapshot,
            optimized=optimized_snapshot,
            usage_event=usage_event,
            notes={
                "project_id": project_id,
                "model_name": named_model,
                "policy": policy_payload,
                "cache_hit": bool(cached_profile),
                "fingerprint_signature": fingerprint_signature,
                "cache_signature": cache_signature,
                "run_id": run_id,
                "optimization_context": optimization_context,
                "usage_event_payload": usage_payload,
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
        repeats: int,
        warmup: int,
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
        target_device = _infer_module_device(module)
        try:
            if target_device is not None and hasattr(measured_tensor, "device") and measured_tensor.device != target_device:
                measured_tensor = measured_tensor.to(target_device)
        except Exception:
            pass
        warmup_iters = max(0, int(warmup))
        repeat_iters = max(1, int(repeats))
        with torch_mod.inference_mode():
            for _ in range(warmup_iters):
                try:
                    module(measured_tensor)
                    self._sync_device(torch_mod, getattr(measured_tensor, "device", None))
                except Exception:
                    break

            for _ in range(repeat_iters):
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

    def _build_usage_payload(
        self,
        *,
        project_id: str,
        run_id: str,
        named_model: str,
        baseline_snapshot: OptimizationSnapshot,
        optimized_snapshot: OptimizationSnapshot,
        usage_event: Optional[UsageEvent],
        policy: Optional[OptimizationPolicy],
        fingerprint: Optional[Mapping[str, Any]],
        fingerprint_signature: Optional[str],
        optimization_context: Mapping[str, Any],
    ) -> Dict[str, Any]:
        now_ts = int(time.time())
        gpu_payload = {}
        workload_payload = {}
        if fingerprint:
            gpu_payload = dict(fingerprint.get("gpu", {}))
            workload_payload = {
                "framework": fingerprint.get("framework", {}).get("framework"),
                "model": fingerprint.get("model_name"),
                "precision": optimization_context.get("precision")
                if isinstance(optimization_context, dict)
                else None,
            }
            input_sig = fingerprint.get("input_signature", {})
            if isinstance(input_sig, dict):
                workload_payload["input_shape"] = input_sig.get("shape")
                workload_payload["dtype"] = input_sig.get("dtype")
                workload_payload["device"] = input_sig.get("device")

        baseline_metrics = {
            "tokens_per_s": baseline_snapshot.tokens_per_sec,
            "latency_ms": baseline_snapshot.latency_ms,
            "gpu_util": baseline_snapshot.gpu_utilization,
        }
        optimized_metrics = {
            "tokens_per_s": optimized_snapshot.tokens_per_sec,
            "latency_ms": optimized_snapshot.latency_ms,
            "gpu_util": optimized_snapshot.gpu_utilization,
        }

        if usage_event:
            uplift_pct = usage_event.performance_uplift_pct
            gpu_hours = usage_event.gpu_hours_after
            tokens = usage_event.tokens_processed
        else:
            uplift_pct = (
                (optimized_snapshot.tokens_per_sec - baseline_snapshot.tokens_per_sec)
                / max(baseline_snapshot.tokens_per_sec, 1e-6)
            ) * 100.0
            baseline_seconds = baseline_snapshot.latency_ms / 1000.0
            optimized_seconds = optimized_snapshot.latency_ms / 1000.0
            gpu_hours = max(optimized_seconds, 0.0) / 3600.0
            tokens = optimized_snapshot.tokens_processed or baseline_snapshot.tokens_processed

        computed = {
            "uplift_pct": uplift_pct,
            "gpu_hours": gpu_hours,
            "tokens": tokens,
        }

        plan = {
            "objective": policy.plan_objective if policy else "throughput",
            "sample_rate": policy.telemetry_sample_rate if policy else 1.0,
        }

        payload: Dict[str, Any] = {
            "project_id": project_id,
            "run_id": run_id,
            "event": "usage",
            "ts": now_ts,
            "model_name": named_model,
            "gpu": gpu_payload,
            "workload": workload_payload,
            "metrics": {
                "baseline": baseline_metrics,
                "optimized": optimized_metrics,
            },
            "computed": computed,
            "plan": plan,
            "context": dict(optimization_context),
            "sig": fingerprint_signature,
        }
        if usage_event:
            payload["billing"] = {
                "gpu_hours_before": usage_event.gpu_hours_before,
                "gpu_hours_after": usage_event.gpu_hours_after,
                "gpu_hours_saved": usage_event.gpu_hours_saved,
                "usage_charge": usage_event.usage_charge,
                "success_fee": usage_event.success_fee,
                "total_billable": usage_event.total_billable,
                "currency": usage_event.currency,
            }
        if fingerprint:
            payload["fingerprint"] = dict(fingerprint)
        return payload

    def _schedule_auto_retrain(
        self,
        model: Any,
        tensor: "torch.Tensor",
        torch_mod: "torch",
        policy_payload: Mapping[str, Any],
    ) -> None:
        def _runner() -> None:
            try:
                time.sleep(0.1)
                LOGGER.info(
                    "Auto-retrain placeholder executed for policy %s",
                    policy_payload.get("policy_id"),
                )
            except Exception:
                LOGGER.debug("Auto-retrain job failed", exc_info=True)

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()


__all__ = [
    "OptimizationSnapshot",
    "RuntimeOptimizationAgent",
    "RuntimeOptimizationResult",
]
