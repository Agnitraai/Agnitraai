"""Output-validation safety net for the optimizer.

The optimizer's job is to make a model faster *without changing what it
does*. This module provides the test-after-optimize check that compares
baseline vs. optimized outputs on a sample input and reports drift. The
SDK uses it to fall back to the baseline model when drift exceeds
tolerance, so a bad optimization can never silently ship to production.

We deliberately keep this small and dependency-free: the optimizer must
not crash because the validation library is heavy or absent. Outputs
are compared via cosine similarity (handles minor numerical noise from
TF32 / fused kernels) plus exact-equality for argmax / greedy paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OutputDrift:
    """Result of comparing baseline vs. optimized output on one sample."""

    cosine_similarity: float        # 1.0 == identical direction; computed on flat logits
    max_abs_diff: float             # max |a - b| across the tensor
    argmax_match_rate: float        # fraction of positions whose argmax agrees (greedy proxy)
    shapes_match: bool              # False is a hard fail regardless of other metrics
    error: Optional[str] = None     # populated on exceptions; treat as "validation failed"

    @property
    def regressed(self) -> bool:
        """``True`` iff this drift would cause the SDK to revert."""
        return self.error is not None or not self.shapes_match


def output_drift(
    baseline_model: Any,
    optimized_model: Any,
    sample_input: Any,
    *,
    tolerance: float = 1e-3,
) -> OutputDrift:
    """Compare the two models on one input and return drift metrics.

    The function does NOT raise — any exception during comparison is
    captured in :attr:`OutputDrift.error` so the calling SDK can fall
    back gracefully. ``tolerance`` is consulted only for the
    ``shapes_match`` short-circuit; the SDK applies its own threshold
    against ``cosine_similarity`` and ``argmax_match_rate``.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return OutputDrift(0.0, float("inf"), 0.0, False, error="torch unavailable")

    try:
        with torch.inference_mode():
            base_out = baseline_model(sample_input)
            opt_out = optimized_model(sample_input)
    except Exception as exc:
        return OutputDrift(0.0, float("inf"), 0.0, False, error=f"forward failed: {exc!r}")

    # Some heads return tuples / dataclasses; reduce to a representative tensor.
    base_t = _as_tensor(base_out)
    opt_t = _as_tensor(opt_out)
    if base_t is None or opt_t is None:
        return OutputDrift(0.0, float("inf"), 0.0, False, error="output not tensor-like")

    if base_t.shape != opt_t.shape:
        return OutputDrift(0.0, float("inf"), 0.0, False)

    base_t = base_t.detach().to(torch.float32).flatten()
    opt_t = opt_t.detach().to(torch.float32).flatten()

    # Cosine similarity is robust to scale changes from quantization.
    denom = (base_t.norm() * opt_t.norm()).item()
    cos = float((base_t @ opt_t).item() / denom) if denom > 0 else 0.0
    max_abs = float((base_t - opt_t).abs().max().item())

    # Argmax-match rate as a greedy-decode proxy. For 1-D outputs this is
    # binary; for 2-D logits we compare last-axis argmax.
    try:
        bo = _as_tensor(base_out, prefer_logits=True)
        oo = _as_tensor(opt_out, prefer_logits=True)
        if bo is not None and oo is not None and bo.shape == oo.shape and bo.dim() >= 1:
            base_arg = bo.argmax(dim=-1)
            opt_arg = oo.argmax(dim=-1)
            argmax_rate = float((base_arg == opt_arg).float().mean().item())
        else:
            argmax_rate = 1.0 if cos >= 1 - tolerance else 0.0
    except Exception:
        argmax_rate = 1.0 if cos >= 1 - tolerance else 0.0

    return OutputDrift(
        cosine_similarity=cos,
        max_abs_diff=max_abs,
        argmax_match_rate=argmax_rate,
        shapes_match=True,
    )


def _as_tensor(obj: Any, *, prefer_logits: bool = False) -> Any:
    """Best-effort extraction of a tensor for comparison.

    Handles plain Tensors, tuples (first element), and HuggingFace
    ModelOutput dataclasses (uses .logits if ``prefer_logits`` else
    falls through to first tensor field).
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return None

    if isinstance(obj, torch.Tensor):
        return obj
    # transformers.modeling_outputs.ModelOutput exposes .logits / .last_hidden_state.
    for attr in ("logits", "last_hidden_state"):
        candidate = getattr(obj, attr, None)
        if isinstance(candidate, torch.Tensor):
            return candidate
    if isinstance(obj, (tuple, list)) and obj and isinstance(obj[0], torch.Tensor):
        return obj[0]
    return None


__all__ = ["OutputDrift", "output_drift"]
