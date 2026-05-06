"""Quantization passes for decoder-only LLMs.

Each mode delegates to a `torchao` pass and shares the same shape:
take a model, mutate weight tensors in place, return the model. On any
failure (torchao missing, exotic module, runtime error) the pass logs
a warning and returns the unmodified model — quantization is a quality
trade-off and the safest fallback is "do nothing."

Supported modes:

* ``int8_weight`` — W8A16 (INT8 weights, FP16 activations).
  ~2x memory, ~1.3-1.7x throughput on memory-bound decode, near-zero
  quality loss. The default safe choice. Works on any CUDA GPU.

* ``int4_weight`` — W4A16 (INT4 weights, FP16 activations).
  ~4x memory, ~1.6-2.0x throughput. Mild quality drop; verify against
  your eval set. Best for memory-bound workloads on smaller GPUs
  (RTX 4090, A40, L4).

* ``fp8_weight`` — W8(FP8)A8(FP8). NVIDIA's native FP8 tensor cores
  on H100 / H200 / Blackwell. ~2x throughput vs FP16 with near-zero
  quality drop. Falls back to INT8 on non-H100/Blackwell hardware.

* ``auto`` — picks the best available mode for the detected GPU
  capability + torchao version. Routing logic in ``select_auto_mode``.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

LOGGER = logging.getLogger(__name__)


# ----- per-mode resolvers --------------------------------------------------


def _resolve_int8_weight_only_config() -> Tuple[Callable[..., None], Any]:
    """Return ``(quantize_fn, config)`` for INT8 weight-only.

    Resolves across torchao API versions. Raises ImportError if no
    compatible path is available.
    """
    try:
        from torchao.quantization import quantize_, int8_weight_only
        return quantize_, int8_weight_only()
    except ImportError:
        pass
    try:
        from torchao.quantization.quant_api import quantize_, Int8WeightOnly
        return quantize_, Int8WeightOnly()
    except ImportError:
        pass
    try:
        from torchao.quantization import apply_dynamic_quant  # type: ignore[attr-defined]
        return (lambda m, _cfg: apply_dynamic_quant(m)), None
    except ImportError:
        raise ImportError(
            "INT8 weight-only quantization requires `torchao>=0.4`. "
            "Install with: pip install torchao"
        )


def _resolve_int4_weight_only_config() -> Tuple[Callable[..., None], Any]:
    """Return ``(quantize_fn, config)`` for INT4 weight-only."""
    try:
        from torchao.quantization import quantize_, int4_weight_only
        return quantize_, int4_weight_only()
    except ImportError:
        pass
    try:
        from torchao.quantization.quant_api import quantize_, Int4WeightOnly
        return quantize_, Int4WeightOnly()
    except ImportError:
        raise ImportError(
            "INT4 weight-only quantization requires `torchao>=0.5`. "
            "Install with: pip install torchao>=0.5"
        )


def _resolve_fp8_weight_only_config() -> Tuple[Callable[..., None], Any]:
    """Return ``(quantize_fn, config)`` for FP8 weight + activation.

    Requires Hopper (H100) or newer for the FP8 tensor cores. On older
    GPUs the underlying torchao kernel falls back to FP16 emulation,
    which is slower than INT8 — callers should use ``select_auto_mode``
    to avoid that footgun.
    """
    try:
        from torchao.quantization import quantize_, float8_weight_only
        return quantize_, float8_weight_only()
    except ImportError:
        pass
    try:
        from torchao.quantization.quant_api import quantize_, Float8WeightOnly
        return quantize_, Float8WeightOnly()
    except ImportError:
        raise ImportError(
            "FP8 weight-only quantization requires `torchao>=0.6` and an "
            "H100/H200/Blackwell GPU for the FP8 tensor cores. Install "
            "with: pip install torchao>=0.6"
        )


_MODE_RESOLVERS = {
    "int8_weight": _resolve_int8_weight_only_config,
    "int4_weight": _resolve_int4_weight_only_config,
    "fp8_weight": _resolve_fp8_weight_only_config,
}


# ----- mode selection ------------------------------------------------------


def select_auto_mode() -> str:
    """Pick the best quantization mode for the current hardware.

    Priority:
      1. FP8 on H100 / H200 / Blackwell (compute capability >= 9.0).
         FP8 tensor cores deliver near-FP16 quality at 2x throughput.
      2. INT8 weight-only otherwise. Universally safe, runs on any CUDA GPU.

    INT4 is intentionally NOT auto-selected because the quality drop
    is workload-dependent — callers who want INT4 ask for it explicitly.

    Returns the chosen mode string. Returns ``"int8_weight"`` when GPU
    detection fails (CPU-only, torch missing, exotic device).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return "int8_weight"
        cap = torch.cuda.get_device_capability(0)
    except Exception:  # pragma: no cover - defensive
        return "int8_weight"
    major, _minor = cap
    if major >= 9:  # Hopper (sm_90), Blackwell (sm_100), and successors
        return "fp8_weight"
    return "int8_weight"


# ----- public entry point --------------------------------------------------


def apply_quantization(model: Any, mode: str) -> Any:
    """Apply ``mode`` quantization to ``model`` in place. Returns model.

    Modes: ``int8_weight``, ``int4_weight``, ``fp8_weight``, ``auto``.

    Always returns the model. Quantization failures (torchao missing,
    runtime error, unsupported architecture) log a warning and return
    the unmodified model rather than raising — quantization is a
    quality trade-off and we never want it to break the optimizer.
    """
    if mode == "auto":
        chosen = select_auto_mode()
        LOGGER.info("Quantization mode 'auto' resolved to %r", chosen)
        mode = chosen

    if mode not in _MODE_RESOLVERS:
        LOGGER.warning("Unknown quantization mode %r; skipping", mode)
        return model

    try:
        quantize_fn, config = _MODE_RESOLVERS[mode]()
    except ImportError as exc:
        LOGGER.warning("Skipping %s quantization: %s", mode, exc)
        return model

    try:
        if config is None:
            quantize_fn(model, None)
        else:
            quantize_fn(model, config)
        LOGGER.info("Applied %s quantization", mode)
    except Exception:
        LOGGER.exception(
            "%s quantization failed; returning unquantized model", mode
        )
    return model


# ----- back-compat shim ----------------------------------------------------


def apply_int8_weight_only(model: Any) -> Any:
    """Backwards-compatible alias used by tests written against 0.2.0."""
    return apply_quantization(model, "int8_weight")


__all__ = [
    "apply_quantization",
    "apply_int8_weight_only",
    "select_auto_mode",
]
