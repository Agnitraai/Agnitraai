"""Architecture-specific optimizers (ring 1: decoder-only LLMs).

The wedge says Agnitra is the inference optimizer for *decoder-only LLMs*
— Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, and their fine-tunes.
This package encodes that commitment in code.

Public surface:

* :func:`agnitra.optimizers.detection.detect_architecture` — returns the
  HuggingFace ``model_type`` string (or ``None``) for any
  :class:`torch.nn.Module`.
* :data:`agnitra.optimizers.registry.SUPPORTED_DECODER_LM_TYPES` — the
  canonical list of ring-1 architectures.
* :func:`agnitra.optimizers.registry.is_supported` — ``True`` iff a
  ``model_type`` is in the ring-1 set.

Architecture-specialist modules (``decoder_lm/llama.py``, etc.) land in
later steps. For now this package's only job is honest scoping: tell
``agnitra.optimize`` whether to run the optimizer or pass through.
"""
from __future__ import annotations

from agnitra.optimizers.detection import detect_architecture
from agnitra.optimizers.registry import (
    SUPPORTED_DECODER_LM_TYPES,
    is_supported,
)

__all__ = [
    "detect_architecture",
    "is_supported",
    "SUPPORTED_DECODER_LM_TYPES",
]
