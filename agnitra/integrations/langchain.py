"""LangChain integration for Agnitra.

LangChain agents call the LLM hundreds of times per task — every
prompt-tool-response loop hits the model. Optimization compounds: a
1.5x speedup on a single forward pass becomes a 1.5x reduction in
wall-clock latency across the entire agent task.

This module wraps the existing ``agnitra.optimize`` SDK so a single
function call swaps out the model inside an existing LangChain LLM
without changing the rest of the agent code.

Supported LLM types (auto-detected):

* ``langchain_huggingface.HuggingFacePipeline`` (the modern path)
* ``langchain_community.llms.huggingface_pipeline.HuggingFacePipeline``
  (older path, still common in tutorials)
* ``langchain.llms.huggingface_pipeline.HuggingFacePipeline``
  (legacy path, deprecated but in the wild)

For other LLM types the function logs a warning and returns the LLM
unchanged. PRs welcome to extend coverage.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agnitra.integrations.huggingface import wrap_model

LOGGER = logging.getLogger(__name__)


def _extract_pipeline(llm: Any) -> Optional[Any]:
    """Return the inner ``transformers.Pipeline`` if ``llm`` wraps one.

    LangChain's HuggingFacePipeline class stores it as ``.pipeline``;
    other wrappers may use ``._pipeline`` or ``.client``. Try common
    attribute names; return None when nothing matches.
    """
    for attr in ("pipeline", "_pipeline", "client", "hf_pipeline"):
        candidate = getattr(llm, attr, None)
        if candidate is not None and hasattr(candidate, "model"):
            return candidate
    return None


def optimize_llm(
    llm: Any,
    *,
    agnitra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Replace the model inside a LangChain LLM with the optimized version.

    The LLM instance is mutated in place AND returned, so callers can
    write either of these patterns:

        optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 512)})
        # or
        llm = optimize_llm(llm, agnitra_kwargs={...})

    ``agnitra_kwargs`` is forwarded to ``agnitra.optimize`` — callers
    who want INT8 quantization for an agent workload pass
    ``{"input_shape": (1, 512), "quantize": "int8_weight"}``.
    """
    pipeline = _extract_pipeline(llm)
    if pipeline is None:
        LOGGER.warning(
            "optimize_llm: LLM type %s is not a recognized HuggingFace "
            "wrapper; returning unchanged. Open an issue to add support.",
            type(llm).__name__,
        )
        return llm

    optimized = wrap_model(pipeline.model, agnitra_kwargs=agnitra_kwargs)
    pipeline.model = optimized
    LOGGER.info(
        "optimize_llm: swapped model inside %s (pipeline=%s)",
        type(llm).__name__,
        type(pipeline).__name__,
    )
    return llm


__all__ = ["optimize_llm"]
