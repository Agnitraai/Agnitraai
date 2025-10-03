"""Public SDK facade providing high-level helpers."""
from __future__ import annotations

from typing import Any, Optional, Sequence, TYPE_CHECKING

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

__all__ = [
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
    "apply_tuning_preset",
]


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
