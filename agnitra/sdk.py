"""Public SDK facade providing high-level helpers."""
from __future__ import annotations

import logging
import os
import time
from contextlib import suppress
from typing import Any, Dict, Mapping, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - imported for type checkers only
    import torch
    from torch import Tensor, nn
else:  # pragma: no cover - runtime fallbacks when torch absent
    Tensor = Any  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]

from agnitra._sdk import (
    FXNodePatch,
    ForwardHookPatch,
    IRExtractor,
    LLMOptimizer,
    PatchLog,
    RLAgent,
    RuntimePatchReport,
    RuntimePatcher,
    Telemetry,
    CodexGuidedAgent,
    KernelGenerator,
    apply_tuning_preset,
)
from agnitra._sdk.optimizer import optimize_model as _optimize_model
from agnitra.core.licensing import LicenseManager, LicenseValidationError
from agnitra.core.metering import UsageEvent, UsageMeter
from agnitra.core.runtime import (
    OptimizationSnapshot,
    RuntimeOptimizationAgent,
    RuntimeOptimizationResult,
)
from agnitra.core.runtime.cache import CachedProfile, OptimizationCache
from agnitra.core.runtime.control_plane import ControlPlaneClient, OptimizationPolicy
from agnitra.core.runtime.fingerprint import fingerprint_signature, fingerprint_workload
from agnitra.core.runtime.telemetry_client import TelemetryClient, TelemetryConfig

__all__ = [
    "optimize",
    "optimize_model",
    "resolve_input_tensor",
    "Telemetry",
    "IRExtractor",
    "LLMOptimizer",
    "RLAgent",
    "CodexGuidedAgent",
    "KernelGenerator",
    "FXNodePatch",
    "ForwardHookPatch",
    "PatchLog",
    "RuntimePatchReport",
    "RuntimePatcher",
    "OptimizationSnapshot",
    "RuntimeOptimizationAgent",
    "RuntimeOptimizationResult",
    "UsageEvent",
    "UsageMeter",
    "apply_tuning_preset",
]

LOGGER = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _prepare_license_manager(
    license_manager: Optional[LicenseManager],
    *,
    require: bool = False,
) -> Optional[LicenseManager]:
    manager = license_manager or LicenseManager()
    try:
        manager.load()
    except LicenseValidationError as exc:
        if require:
            raise
        LOGGER.debug("License unavailable; continuing without enforcement: %s", exc)
        return None
    return manager


def resolve_input_tensor(
    model: "nn.Module",
    input_tensor: Optional["Tensor"] = None,
    *,
    input_shape: Optional[Sequence[int]] = None,
    device: Optional["torch.device"] = None,
) -> "Tensor":
    """Public helper mirroring the SDK input preparation logic."""

    return _prepare_input(model, input_tensor, input_shape, device)


def _prepare_input(
    model: "nn.Module",
    input_tensor: Optional["Tensor"],
    input_shape: Optional[Sequence[int]],
    device: Optional["torch.device"],
) -> "Tensor":
    """Resolve the tensor used for optimization telemetry collection."""

    torch_mod = _require_torch()

    if input_tensor is not None:
        return input_tensor

    if input_shape is not None:
        return torch_mod.randn(*input_shape, device=device)

    example = getattr(model, "example_input_array", None)
    if example is not None:
        if isinstance(example, torch_mod.Tensor):
            return example.to(device=device) if device else example
        if isinstance(example, (list, tuple)) and example and isinstance(example[0], torch_mod.Tensor):
            return example[0].to(device=device) if device else example[0]

    raise ValueError(
        "optimize_model requires either input_tensor or input_shape. "
        "Set input_shape=(...) or supply a ready tensor."
    )


def optimize_model(
    model: "nn.Module",
    input_tensor: Optional["Tensor"] = None,
    *,
    input_shape: Optional[Sequence[int]] = None,
    device: Optional["torch.device"] = None,
    enable_rl: bool = True,
) -> "nn.Module":
    """Optimize ``model`` using Agnitra's pipeline.

    Parameters
    ----------
    model:
        PyTorch module to optimize.
    input_tensor:
        Optional tensor describing the input batch. When omitted, provide
        ``input_shape`` or define ``model.example_input_array``.
    input_shape:
        Convenience helper to synthesize a random tensor when ``input_tensor`` is
        omitted. The tensor is sampled from a standard normal distribution.
    device:
        When set, the generated tensor is allocated on the given device. Existing
        tensors are moved best-effort.
    enable_rl:
        Toggles the PPO-based reinforcement learning stage.

    Returns
    -------
    nn.Module
        The optimized module or the original instance when optimization fails.
    """

    torch_mod = _require_torch()

    tensor = resolve_input_tensor(model, input_tensor, input_shape=input_shape, device=device)
    if device is not None and isinstance(tensor, torch_mod.Tensor) and tensor.device != device:
        tensor = tensor.to(device)

    return _optimize_model(model, tensor, enable_rl=enable_rl)


def optimize(
    model: "nn.Module",
    input_tensor: Optional["Tensor"] = None,
    *,
    input_shape: Optional[Sequence[int]] = None,
    device: Optional["torch.device"] = None,
    enable_rl: bool = True,
    project_id: str = "default",
    model_name: Optional[str] = None,
    usage_meter: Optional[UsageMeter] = None,
    repeats: int = 10,
    warmup: int = 3,
    rate_per_gpu_hour: float = 2.5,
    success_margin_pct: float = 0.2,
    metadata: Optional[Mapping[str, Any]] = None,
    telemetry_client: Optional[TelemetryClient] = None,
    control_plane_client: Optional[ControlPlaneClient] = None,
    optimization_cache: Optional[OptimizationCache] = None,
    offline: bool = False,
    license_manager: Optional[LicenseManager] = None,
    require_license: bool = False,
    license_seat: Optional[str] = None,
    license_org_id: Optional[str] = None,
) -> RuntimeOptimizationResult:
    """Optimize ``model`` and return a metered runtime optimization report.

    Parameters
    ----------
    offline:
        When ``True`` the SDK avoids network calls and requires the license to
        grant the ``offline`` feature.
    license_manager:
        Optional :class:`~agnitra.core.licensing.LicenseManager` instance used to
        enforce enterprise/offline entitlement and per-GPU tracking.
    require_license:
        Force license validation even when ``offline`` is ``False``.
    license_seat:
        Optional seat identifier; defaults to the workload fingerprint
        signature.
    license_org_id:
        Overrides the organisation identifier recorded for per-GPU licensing.
    """

    torch_mod = _require_torch()
    tensor = resolve_input_tensor(model, input_tensor, input_shape=input_shape, device=device)
    if device is not None and isinstance(tensor, torch_mod.Tensor) and tensor.device != device:
        tensor = tensor.to(device)

    metadata_map: Dict[str, Any] = dict(metadata or {})
    metadata_map.setdefault("project_id", project_id)

    require_flag = require_license or _coerce_bool(os.environ.get("AGNITRA_REQUIRE_LICENSE", "0"))
    license_context: Dict[str, Any] = {}
    license_manager_instance: Optional[LicenseManager] = None
    if offline or require_flag or license_manager is not None or os.environ.get("AGNITRA_LICENSE_PATH"):
        try:
            license_manager_instance = _prepare_license_manager(
                license_manager,
                require=offline or require_flag,
            )
        except LicenseValidationError as exc:
            LOGGER.error("License validation failed: %s", exc)
            raise
    if license_manager_instance and offline:
        try:
            license_manager_instance.ensure_feature("offline")
        except LicenseValidationError as exc:
            LOGGER.error("Offline mode requires license entitlement: %s", exc)
            raise

    if telemetry_client is not None:
        telemetry_client_instance: Optional[TelemetryClient] = telemetry_client
    elif offline:
        telemetry_client_instance = None
    else:
        telemetry_client_instance = TelemetryClient(TelemetryConfig())

    if control_plane_client is not None:
        control_client: Optional[ControlPlaneClient] = control_plane_client
    elif offline:
        control_client = None
    else:
        control_client = ControlPlaneClient()

    cache = optimization_cache or OptimizationCache()
    cache.clear_expired()

    fingerprint_obj = fingerprint_workload(
        model,
        tensor,
        metadata={
            "project_id": project_id,
            "model_name": model_name or getattr(model, "__class__", type(model)).__name__,
        },
    )
    fingerprint_dict = fingerprint_obj.to_dict()
    fingerprint_sig = fingerprint_signature(fingerprint_dict)

    seat_id = license_seat or fingerprint_sig
    seat_checked_out = False

    try:
        if license_manager_instance:
            try:
                license_manager_instance.checkout_seat(seat_id)
                seat_checked_out = True
                license_context["seat_id"] = seat_id
                license_context["license"] = license_manager_instance.to_dict()
            except LicenseValidationError as exc:
                if offline or require_flag:
                    raise
                LOGGER.warning("License seat checkout failed; continuing without enforcement: %s", exc)
                license_manager_instance = None
            except Exception as exc:
                if offline or require_flag:
                    raise LicenseValidationError(str(exc))
                LOGGER.warning("Unexpected license checkout error; disabling enforcement: %s", exc)
                license_manager_instance = None

        if control_client is not None:
            try:
                policy = control_client.fetch_policy(project_id, fingerprint_dict)
            except Exception:
                LOGGER.exception("Failed to fetch optimization policy; using default")
                policy = OptimizationPolicy()
        else:
            policy = OptimizationPolicy()

        cached_profile: Optional[CachedProfile] = cache.lookup(fingerprint_sig)
        if cached_profile:
            LOGGER.info("Optimization cache hit for signature %s", fingerprint_sig[:8])

        policy_repeats = policy.calibration_iterations if policy else repeats
        policy_warmup = policy.calibration_warmup if policy else warmup
        calibration_repeats = max(repeats, policy_repeats)
        calibration_warmup = max(warmup, policy_warmup)

        agent = RuntimeOptimizationAgent(
            usage_meter=usage_meter,
            repeats=calibration_repeats,
            warmup=calibration_warmup,
            rate_per_gpu_hour=rate_per_gpu_hour,
            success_margin_pct=success_margin_pct,
        )

        start_time = time.time()
        result = agent.optimize(
            model,
            tensor,
            project_id=project_id,
            model_name=model_name,
            enable_rl=enable_rl,
            metadata={
                **metadata_map,
                "fingerprint_signature": fingerprint_sig,
            },
            policy=policy,
            cached_profile=cached_profile,
            telemetry_client=telemetry_client_instance,
            fingerprint=fingerprint_dict,
            fingerprint_signature=fingerprint_sig,
            cache_signature=fingerprint_sig,
        )
        duration = time.time() - start_time

        optimization_context = result.notes.get("optimization_context", {}) if isinstance(result.notes, dict) else {}
        applied_preset = optimization_context.get("applied_preset")
        usage_payload = result.notes.get("usage_event_payload", {})
        improved = result.optimized.latency_ms <= result.baseline.latency_ms

        cache_payload = {
            "applied_preset": applied_preset,
            "policy_id": getattr(policy, "policy_id", None),
            "baseline_latency_ms": result.baseline.latency_ms,
            "optimized_latency_ms": result.optimized.latency_ms,
            "tokens_processed": result.optimized.tokens_processed,
            "usage_event": usage_payload.get("billing"),
            "cached_at": time.time(),
            "duration_ms": duration * 1000.0,
        }
        cache_stored = False
        ttl_seconds = policy.cache_ttl_seconds if policy else 86400
        if applied_preset or improved:
            cache.store(fingerprint_sig, cache_payload, ttl_seconds=ttl_seconds)
            cache_stored = True

        cache_info = {
            "signature": fingerprint_sig,
            "hit": bool(cached_profile),
            "stored": cache_stored,
            "ttl_seconds": ttl_seconds,
        }
        if isinstance(result.notes, dict):
            result.notes["cache_info"] = cache_info
            result.notes["fingerprint"] = fingerprint_dict

        if license_manager_instance:
            with suppress(LicenseValidationError):
                usage_record = license_manager_instance.register_gpu_run(
                    org_id=license_org_id or project_id
                )
                license_context["gpu_usage"] = usage_record

        if license_context and isinstance(result.notes, dict):
            result.notes.setdefault("license", license_context)

        if telemetry_client is None and telemetry_client_instance is not None:
            telemetry_client_instance.close()

        return result
    finally:
        if license_manager_instance and seat_checked_out:
            with suppress(Exception):
                license_manager_instance.release_seat(seat_id)


_TORCH: Optional[Any] = None


def _require_torch() -> "torch":
    """Import ``torch`` lazily to keep optional dependency lightweight."""

    global _TORCH
    if _TORCH is None:
        try:
            import torch as torch_mod  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - torch absent at runtime
            raise RuntimeError("PyTorch is required to optimize models") from exc
        _TORCH = torch_mod
    return _TORCH
