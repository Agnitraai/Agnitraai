"""LlamaIndex integration for Agnitra AI.

LlamaIndex's RAG and agent flows hit the LLM repeatedly: query
rewriting, response synthesis, sub-question decomposition, evaluator
calls. Same compounding effect as LangChain agents — a 1.5x model
speedup is a 1.5x reduction in pipeline wall time.

Supported LLM types (auto-detected):

* ``llama_index.llms.huggingface.HuggingFaceLLM`` (current path,
  ``llama-index-llms-huggingface`` package)
* ``llama_index.legacy.llms.huggingface.HuggingFaceLLM`` (legacy)

For other LLM types the function logs a warning and returns the LLM
unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from agnitra.integrations.huggingface import wrap_model

LOGGER = logging.getLogger(__name__)


def _extract_inner_model(llm: Any) -> Optional[Any]:
    """Return the underlying ``nn.Module`` from a LlamaIndex LLM wrapper.

    LlamaIndex stores the HF model as ``._model``; some adapters use
    ``.model``. Try both and return None when nothing matches.
    """
    for attr in ("_model", "model"):
        candidate = getattr(llm, attr, None)
        # Sanity check: must be a module-like object that has parameters.
        if candidate is not None and hasattr(candidate, "parameters"):
            return candidate
    return None


def optimize_llm(
    llm: Any,
    *,
    agnitra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Replace the model inside a LlamaIndex LLM with the optimized version.

    Mutates the LLM in place AND returns it — both call patterns work::

        optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 512)})
        llm = optimize_llm(llm, agnitra_kwargs={...})

    ``agnitra_kwargs`` is forwarded to ``agnitra.optimize``.
    """
    inner = _extract_inner_model(llm)
    if inner is None:
        LOGGER.warning(
            "optimize_llm: LLM type %s does not expose a recognized inner "
            "model attribute; returning unchanged. Open an issue to add support.",
            type(llm).__name__,
        )
        return llm

    optimized = wrap_model(inner, agnitra_kwargs=agnitra_kwargs)
    # Try setting via the same attribute we read from.
    for attr in ("_model", "model"):
        if hasattr(llm, attr) and getattr(llm, attr) is inner:
            try:
                setattr(llm, attr, optimized)
                LOGGER.info(
                    "optimize_llm: swapped %s.%s with optimized model",
                    type(llm).__name__,
                    attr,
                )
                break
            except Exception:  # pragma: no cover - defensive
                continue
    return llm


__all__ = ["optimize_llm"]
