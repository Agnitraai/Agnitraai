"""Ring-1 specialist optimizers for decoder-only LLMs.

Architecture-specific entry points dispatch to the right specialization
based on the ``model_type`` returned by detection. Every specialist
applies a curated, hard-coded sequence of proven passes — TF32, SDPA,
static KV cache, ``torch.compile`` with cudagraphs — plus any
architecture-unique tuning (e.g. fused RoPE for Llama).

Why hard-coded rather than LLM-/RL-guided: the LLM/RL paths in
``agnitra/_sdk/`` are research tools. Production customers need
deterministic, repeatable behavior. The specialists encode the
optimization sequence we *know* works for each architecture; the
research path stays available behind ``enable_rl=True`` for users who
want to explore.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from agnitra.optimizers.decoder_lm import (
    _passes,
    gemma as _gemma,
    llama as _llama,
    mistral as _mistral,
    qwen2 as _qwen2,
)


LOGGER = logging.getLogger(__name__)


# Mapping from HF ``model_type`` to the specialist module that knows
# how to optimize it. Architectures not in this map but in
# ``SUPPORTED_DECODER_LM_TYPES`` fall through to the shared
# ``decoder_lm_generic`` pipeline (still better than the legacy
# LLM/RL path, just not architecture-tuned).
_DISPATCH: Dict[str, Callable[..., Any]] = {
    "llama": _llama.optimize,
    "mistral": _mistral.optimize,
    "mixtral": _mistral.optimize,  # same architectural family
    "qwen2": _qwen2.optimize,
    "qwen2_moe": _qwen2.optimize,
    "gemma": _gemma.optimize,
    "gemma2": _gemma.optimize,
}


def optimize_decoder_lm(
    model: Any,
    *,
    model_type: str,
    sample_input: Any,
    enable_compile: bool = True,
) -> Any:
    """Apply the architecture-appropriate specialist sequence.

    Returns the optimized ``nn.Module``. The SDK wraps this call with
    its measurement, validation, and caching layers — the specialist
    only has to produce a correct, faster module.

    ``enable_compile=False`` is for tests: ``torch.compile`` is slow at
    first invocation and we don't want every unit test to pay that
    cost. Production callers should always leave this on.
    """
    handler = _DISPATCH.get(model_type, _generic_decoder_lm)
    LOGGER.info(
        "decoder_lm specialist: %s (handler=%s)", model_type, handler.__module__
    )
    return handler(model, sample_input=sample_input, enable_compile=enable_compile)


def _generic_decoder_lm(model, *, sample_input, enable_compile: bool = True):
    """Fallback for ring-1 architectures without a tuned specialist yet.

    Applies only the universally-safe passes. Architecture-specific
    fusions (RoPE, RMSNorm) are skipped.
    """
    return _passes.apply_universal(model, sample_input=sample_input, enable_compile=enable_compile)


__all__ = ["optimize_decoder_lm"]
