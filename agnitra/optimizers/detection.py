"""Architecture detection.

Two signals, in priority order:

1. ``model.config.model_type`` — canonical when available, present on
   every HuggingFace ``transformers`` model.
2. Structural fingerprint — count transformer blocks, attention modules,
   and a final language-modeling head. Catches raw ``torch``-saved
   models that have lost their ``config`` but retain the structural
   pattern Agnitra targets.

Returns the model_type string (which downstream code compares against
:data:`SUPPORTED_DECODER_LM_TYPES`) or ``None`` when no decoder-LM
pattern is detectable. ``None`` is the explicit "out of scope" signal
— the caller should return a pass-through optimization result.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)


_DECODER_LM_HINT_NAMES = (
    "lm_head",
    "embed_tokens",
    "wte",  # GPT-style alias
)


def _config_model_type(model: Any) -> Optional[str]:
    """Read ``model.config.model_type`` if present; else ``None``.

    ``getattr(obj, "attr", default)`` only swallows AttributeError —
    descriptors (like ``@property``) that raise other exceptions
    propagate. Wrap defensively so detection is never the reason an
    ``optimize()`` call crashes.
    """
    try:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return None
        mtype = getattr(cfg, "model_type", None)
    except Exception:  # pragma: no cover - defensive against exotic descriptors
        return None
    if isinstance(mtype, str) and mtype:
        return mtype.lower()
    return None


def _looks_like_decoder_lm(model: Any) -> bool:
    """Best-effort structural test for a decoder-only LM.

    Heuristic: at least one module name in ``_DECODER_LM_HINT_NAMES``
    appears in the module tree, AND there is at least one descendant
    module whose class name contains "Attention". This catches
    transformers' own model classes without requiring a config and
    without false-positiving on (say) a CNN.
    """
    has_lm_head_hint = False
    has_attention = False
    try:
        for name, submodule in model.named_modules():
            short = name.split(".")[-1] if name else ""
            if short in _DECODER_LM_HINT_NAMES:
                has_lm_head_hint = True
            if "attention" in type(submodule).__name__.lower():
                has_attention = True
            if has_lm_head_hint and has_attention:
                return True
    except Exception:  # pragma: no cover - defensive against exotic modules
        return False
    return False


def detect_architecture(model: Any) -> Optional[str]:
    """Return the HuggingFace ``model_type`` for ``model``, else ``None``.

    A non-``None`` return does NOT imply the architecture is in the
    ring-1 supported set — callers must additionally check
    :func:`agnitra.optimizers.registry.is_supported`. Returning
    ``"decoder_lm_generic"`` is the structural-fallback signal used when
    a model looks decoder-shaped but its specific ``model_type`` cannot
    be read.
    """
    mtype = _config_model_type(model)
    if mtype is not None:
        return mtype
    if _looks_like_decoder_lm(model):
        return "decoder_lm_generic"
    return None


__all__ = ["detect_architecture"]
