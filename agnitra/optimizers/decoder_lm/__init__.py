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


# Mapping from HF ``model_type`` to the specialist *module* that knows
# how to optimize it. We resolve ``module.optimize`` at call time
# (rather than capturing the function reference at import) so that
# ``monkeypatch.setattr(llama, "optimize", ...)`` in tests actually
# affects dispatch — capturing the function would freeze it at
# import and ignore the patch.
_DISPATCH = {
    "llama": _llama,
    "mistral": _mistral,
    "mixtral": _mistral,  # same architectural family
    "qwen2": _qwen2,
    "qwen2_moe": _qwen2,
    "gemma": _gemma,
    "gemma2": _gemma,
}


def optimize_decoder_lm(
    model: Any,
    *,
    model_type: str,
    sample_input: Any,
    enable_compile: bool = True,
    quantize: Optional[str] = None,
) -> Any:
    """Apply the architecture-appropriate specialist sequence.

    Returns the optimized ``nn.Module``. The SDK wraps this call with
    its measurement, validation, and caching layers — the specialist
    only has to produce a correct, faster module.

    ``enable_compile=False`` is for tests: ``torch.compile`` is slow at
    first invocation and we don't want every unit test to pay that
    cost. Production callers should always leave this on.

    ``quantize="int8_weight"`` enables INT8 weight-only quantization
    via torchao. This is the optimization that gives Agnitra a real
    speedup over plain HuggingFace + ``torch.compile`` (HF doesn't
    quantize by default). Expected: ~1.3-1.7x throughput on
    memory-bound decode plus 2x memory reduction.
    """
    module = _DISPATCH.get(model_type)
    if module is None:
        handler = _generic_decoder_lm
        handler_label = "decoder_lm._generic_decoder_lm"
    else:
        handler = module.optimize  # late binding — picks up monkeypatched fakes
        handler_label = module.__name__
    LOGGER.info(
        "decoder_lm specialist: %s (handler=%s, quantize=%r)",
        model_type,
        handler_label,
        quantize,
    )
    return handler(
        model,
        sample_input=sample_input,
        enable_compile=enable_compile,
        quantize=quantize,
    )


def _generic_decoder_lm(
    model,
    *,
    sample_input,
    enable_compile: bool = True,
    quantize: Optional[str] = None,
):
    """Fallback for ring-1 architectures without a tuned specialist yet.

    Applies only the universally-safe passes plus optional quantization.
    Architecture-specific fusions (RoPE, RMSNorm) are skipped.
    """
    return _passes.apply_universal(
        model,
        sample_input=sample_input,
        enable_compile=enable_compile,
        quantize=quantize,
    )


__all__ = ["optimize_decoder_lm"]
