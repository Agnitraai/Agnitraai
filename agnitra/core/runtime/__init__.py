"""Runtime patching and tuning utilities."""

from __future__ import annotations

from .agent import OptimizationSnapshot, RuntimeOptimizationAgent, RuntimeOptimizationResult
from .oem import EmbeddedTelemetry, suggest_kernel_config
from .runtime_patcher import (
    FXNodePatch,
    ForwardHookPatch,
    PatchLog,
    RuntimePatchReport,
    RuntimePatcher,
)
from .tuning import apply_tuning_preset

__all__ = [
    "OptimizationSnapshot",
    "RuntimeOptimizationAgent",
    "RuntimeOptimizationResult",
    "FXNodePatch",
    "ForwardHookPatch",
    "PatchLog",
    "RuntimePatchReport",
    "RuntimePatcher",
    "apply_tuning_preset",
    "EmbeddedTelemetry",
    "suggest_kernel_config",
]
