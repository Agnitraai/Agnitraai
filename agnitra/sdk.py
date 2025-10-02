"""Public SDK facade providing high-level helpers."""
from __future__ import annotations

from typing import Any, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import torch
    from torch import Tensor, nn
except Exception:  # pragma: no cover - exercised in environments without torch
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

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

__all__ = [
    "optimize_model",
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
    "apply_tuning_preset",
]


def _prepare_input(
    model: "nn.Module",
    input_tensor: Optional["Tensor"],
    input_shape: Optional[Sequence[int]],
    device: Optional["torch.device"],
) -> "Tensor":
    """Resolve the tensor used for optimization telemetry collection."""

    if torch is None:  # pragma: no cover - torch absent
        raise RuntimeError("PyTorch is required to optimize models")

    if input_tensor is not None:
        return input_tensor

    if input_shape is not None:
        return torch.randn(*input_shape, device=device)

    example = getattr(model, "example_input_array", None)
    if example is not None:
        if isinstance(example, torch.Tensor):
            return example.to(device=device) if device else example
        if isinstance(example, (list, tuple)) and example and isinstance(example[0], torch.Tensor):
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

    if torch is None:  # pragma: no cover - torch absent
        raise RuntimeError("PyTorch is required to optimize models")

    tensor = _prepare_input(model, input_tensor, input_shape, device)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)

    return _optimize_model(model, tensor, enable_rl=enable_rl)
