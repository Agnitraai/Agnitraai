"""Llama specialist (Llama 1/2/3, Code Llama, Llama-3.1, Llama-3.2).

Today this is the universal decoder-LM sequence — TF32, SDPA, static
KV cache, ``torch.compile`` with cudagraphs. Llama-specific Triton
fusions (RoPE + RMSNorm + matmul, fused QKV projection) are tracked
as TODOs; they require GPU access to develop and benchmark and will
land in a follow-up commit once Step 4's first benchmark numbers
identify the highest-leverage fusion target.

The structure exists today so the dispatch layer is correct from day
one and so adding the kernel-level passes is a one-file change.
"""
from __future__ import annotations

import logging
from typing import Any

from agnitra.optimizers.decoder_lm import _passes


LOGGER = logging.getLogger(__name__)


def optimize(model: Any, *, sample_input: Any, enable_compile: bool = True) -> Any:
    """Apply the Llama-tuned optimization sequence.

    Sequence:
      1. Universal decoder-LM passes (TF32, SDPA, static cache, compile)
      2. [TODO] Fused RMSNorm + RoPE Triton kernel
      3. [TODO] Fused QKV projection (combines Wq, Wk, Wv into one matmul)
    """
    LOGGER.info("Llama specialist: applying optimization sequence")
    model = _passes.apply_universal(
        model, sample_input=sample_input, enable_compile=enable_compile
    )
    # Placeholder for Llama-specific fusions. See _passes for the
    # extension points; see benchmarks/llama3_h100/RESULTS.md for
    # whether the universal sequence already meets the wedge target
    # before investing in custom kernels.
    return model


__all__ = ["optimize"]
