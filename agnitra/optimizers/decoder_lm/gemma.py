"""Gemma 1 / 2 specialist.

Gemma uses GeGLU (vs. SwiGLU) in the FFN and applies post-attention
LayerNorm rather than pre-norm — small structural differences from
Llama that the universal sequence handles correctly. Custom Gemma
RMSNorm fusions are a future optimization.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from agnitra.optimizers.decoder_lm import _passes


LOGGER = logging.getLogger(__name__)


def optimize(
    model: Any,
    *,
    sample_input: Any,
    enable_compile: bool = True,
    quantize: Optional[str] = None,
) -> Any:
    LOGGER.info(
        "Gemma specialist: applying optimization sequence (quantize=%r)",
        quantize,
    )
    return _passes.apply_universal(
        model,
        sample_input=sample_input,
        enable_compile=enable_compile,
        quantize=quantize,
    )


__all__ = ["optimize"]
