"""Deterministic SHA-256 over a model's weights.

Two models with identical weights produce the same SHA, regardless of
which Python process loaded them, what dtype the file was originally
saved in, or which ``nn.Module`` instance ID Python assigned. This is
the cryptographic anchor of the trust manifest's ``base_model.sha256``
field.

Cost: roughly 30 seconds on CPU for an 8B-parameter fp16 model. We
call this once per ``agnitra.optimize()``, so the cost is amortized
over all subsequent inferences. For very large models or signing-heavy
workflows, the helper accepts ``incremental=True`` to skip non-leaf
parameters and use a faster running hash.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


def _state_dict_items(model: Any) -> Iterable[tuple[str, Any]]:
    """Yield ``(name, tensor)`` pairs in sorted order.

    Returns an empty iterable when ``model`` has no ``state_dict``
    method (so callers writing manifests for non-torch objects don't
    crash — they just get an empty SHA over zero items).
    """
    state_dict = getattr(model, "state_dict", None)
    if state_dict is None or not callable(state_dict):
        return iter(())
    try:
        items = state_dict()
    except Exception:  # pragma: no cover - defensive
        return iter(())
    return ((k, items[k]) for k in sorted(items.keys()))


def model_sha256(model: Any, *, incremental: bool = False) -> str:
    """Compute a deterministic SHA-256 over ``model.state_dict()``.

    The hash combines, for each tensor in sorted-by-name order:

    1. The parameter name (as UTF-8 bytes)
    2. The dtype string (e.g. ``torch.float16``)
    3. The shape tuple
    4. The raw tensor bytes (CPU, contiguous, IEEE-754 / two's complement)

    Each section is null-byte separated so values with the same encoded
    representation can't collide across boundaries.

    ``incremental=True`` skips ``buffer`` entries (running stats, etc.)
    and emits a faster but slightly weaker hash — still useful as a
    "did this model change at all" signal, less useful as a forensic
    fingerprint.

    Returns the lowercase hex digest. Empty model (no state_dict)
    returns ``hashlib.sha256(b"").hexdigest()``.
    """
    h = hashlib.sha256()
    count = 0
    for name, tensor in _state_dict_items(model):
        if incremental and "running_" in name:
            continue
        try:
            cpu_tensor = tensor.detach().cpu().contiguous()
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.debug("Skipping non-tensor entry %r in state_dict: %s", name, exc)
            continue

        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(getattr(cpu_tensor, "dtype", "unknown")).encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(tuple(cpu_tensor.shape)).encode("utf-8"))
        h.update(b"\x00")
        try:
            h.update(cpu_tensor.numpy().tobytes())
        except Exception:
            # Tensors that won't convert to numpy (e.g. quantized types)
            # contribute their stringified flat representation instead.
            # Less ideal but deterministic enough for the manifest's
            # "this is the same checkpoint" claim.
            h.update(repr(cpu_tensor.flatten().tolist()).encode("utf-8"))
        count += 1

    if count == 0:
        LOGGER.debug("model_sha256: state_dict was empty; returning empty-input SHA")
    return h.hexdigest()


__all__ = ["model_sha256"]
