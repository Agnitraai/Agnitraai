"""Mistral / Mixtral specialist.

Mistral 7B and Mixtral 8x7B share the Llama-style block layout but add
sliding-window attention (SWA) and grouped-query attention (GQA at 7B,
MoE routing at Mixtral). The universal decoder-LM passes are the
right floor; SWA-aware compile and per-expert MoE routing are tracked
as TODOs — they need GPU access to develop and benchmark.
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
    """Apply the Mistral/Mixtral-tuned optimization sequence.

    Sequence:
      1. INT8 weight-only quantization if requested.
      2. Universal decoder-LM passes.
      3. [TODO] SWA-aware static cache sizing (sliding_window vs. full).
      4. [TODO Mixtral only] Per-expert kernel fusion in the MoE FFN.
    """
    LOGGER.info(
        "Mistral specialist: applying optimization sequence (quantize=%r)",
        quantize,
    )
    return _passes.apply_universal(
        model,
        sample_input=sample_input,
        enable_compile=enable_compile,
        quantize=quantize,
    )


__all__ = ["optimize"]
