"""Qwen 2 / 2.5 specialist (covers qwen2 + qwen2_moe).

Qwen 2 uses a Llama-shaped block with a different RoPE base
(``rope_theta=1e6`` for Qwen2.5 vs. 5e5 for Llama-3) and an enlarged
intermediate size. The universal sequence handles both correctly;
custom kernels for Qwen's specific RoPE constant are a future
optimization.
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
        "Qwen2 specialist: applying optimization sequence (quantize=%r)",
        quantize,
    )
    return _passes.apply_universal(
        model,
        sample_input=sample_input,
        enable_compile=enable_compile,
        quantize=quantize,
    )


__all__ = ["optimize"]
