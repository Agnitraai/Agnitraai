"""INT8 weight-only quantization for decoder-only LLMs.

This is the one optimization where HuggingFace + ``torch.compile``
genuinely don't compete: weight-only INT8 halves the memory footprint
and yields ~1.3-1.7x throughput on memory-bound decode, with near-zero
quality loss on Llama-class models.

The implementation delegates to ``torchao``'s
``Int8WeightOnly``/``int8_weight_only`` pass, which:

  * replaces ``nn.Linear`` weights with INT8 storage,
  * keeps activations in FP16/BF16 (W8A16),
  * registers a custom matmul kernel that dequantizes inline,
  * preserves the model's external interface so HF's ``.generate()``
    works unchanged.

torchao's API has shifted across versions. We try the documented import
paths in order and surface a clear error if none match — this keeps
the rest of the optimizer working when torchao isn't installed.
"""
from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)


def _resolve_int8_weight_only_config():
    """Return a torchao quantization config + the apply function.

    Returns a tuple ``(quantize_fn, config)``. The quantize_fn signature
    is ``quantize_fn(model, config) -> None`` (torchao mutates in place).
    """
    # torchao 0.5+ canonical path.
    try:
        from torchao.quantization import quantize_, int8_weight_only
        return quantize_, int8_weight_only()
    except ImportError:
        pass
    # torchao 0.4.x path.
    try:
        from torchao.quantization.quant_api import quantize_, Int8WeightOnly
        return quantize_, Int8WeightOnly()
    except ImportError:
        pass
    # Legacy torchao path.
    try:
        from torchao.quantization import (  # type: ignore[attr-defined]
            apply_dynamic_quant,
        )
        # Fallback: dynamic quant is similar but slightly different
        # quality profile. Use only when modern paths are unavailable.
        return (lambda m, _cfg: apply_dynamic_quant(m)), None
    except ImportError:
        raise ImportError(
            "INT8 weight-only quantization requires `torchao`. "
            "Install with: `pip install torchao`. Tested with torchao>=0.5."
        )


def apply_int8_weight_only(model: Any) -> Any:
    """Convert ``nn.Linear`` weights in ``model`` to INT8 in place.

    Returns the same model instance (for chainability with the other
    pass functions in ``_passes.py``). On any failure, logs a warning
    and returns the unmodified model — quantization is a quality
    trade-off, so the safest fallback is "do nothing."
    """
    try:
        quantize_fn, config = _resolve_int8_weight_only_config()
    except ImportError as exc:
        LOGGER.warning("Skipping INT8 quantization: %s", exc)
        return model

    try:
        if config is None:
            quantize_fn(model, None)
        else:
            quantize_fn(model, config)
        LOGGER.info("Applied INT8 weight-only quantization")
    except Exception:
        LOGGER.exception(
            "INT8 quantization failed; returning unquantized model"
        )
    return model


__all__ = ["apply_int8_weight_only"]
