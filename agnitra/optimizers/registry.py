"""Ring-1 architecture registry.

The single canonical source of truth for "what does Agnitra optimize?"
Anything not in this set is either pass-through (current behavior) or
explicitly out-of-scope (future rings).

Adding an architecture here is a commitment: the optimizer must produce
measurable speedup AND pass the output-validation safety net for that
architecture's representative model. Don't add an entry until both are
true.
"""
from __future__ import annotations

from typing import FrozenSet


# HuggingFace ``config.model_type`` values for decoder-only LLMs that
# share the structural pattern (RMSNorm/LayerNorm + MHA-with-KV-cache +
# SwiGLU/GeGLU FFN + RoPE) Agnitra targets.
#
# When you add an entry, also add the architecture-specialist module
# under ``agnitra/optimizers/decoder_lm/`` and a benchmark column. Keep
# the comment alongside each entry pointing at the canonical model.
SUPPORTED_DECODER_LM_TYPES: FrozenSet[str] = frozenset({
    "llama",        # meta-llama/Meta-Llama-3-{8B,70B}-Instruct
    "mistral",      # mistralai/Mistral-7B-Instruct-v0.3
    "mixtral",      # mistralai/Mixtral-8x7B-Instruct-v0.1
    "qwen2",        # Qwen/Qwen2.5-{7B,14B,32B}-Instruct
    "qwen2_moe",    # Qwen/Qwen2.5-MoE
    "gemma",        # google/gemma-7b
    "gemma2",       # google/gemma-2-9b-it
    "phi",          # microsoft/phi-2
    "phi3",         # microsoft/Phi-3-mini-4k-instruct
    "deepseek_v2",  # deepseek-ai/DeepSeek-V2-Lite
    "olmo",         # allenai/OLMo-7B
    "yi",           # 01-ai/Yi-1.5-9B
    "falcon",       # tiiuae/falcon-7b
})


def is_supported(model_type: str | None) -> bool:
    """Return True iff ``model_type`` is in the ring-1 set.

    ``None`` is treated as unsupported — callers (e.g. detection
    fallback) should return ``None`` when no architecture can be
    determined, and this helper handles that case so call sites don't
    need explicit None-guards.
    """
    return bool(model_type) and model_type in SUPPORTED_DECODER_LM_TYPES


__all__ = ["SUPPORTED_DECODER_LM_TYPES", "is_supported"]
