"""Helpers for using Agnitra with HuggingFace ``accelerate``.

``accelerate`` is the standard way HuggingFace users prepare models for
inference and training across devices. This module exposes one small
helper rather than subclassing ``Accelerator``, because subclassing
forces users to swap their ``Accelerator`` import — and most
``accelerate`` users don't construct ``Accelerator`` themselves; it's
constructed inside ``transformers.Trainer`` and friends.

The clean integration is::

    from accelerate import Accelerator
    from agnitra.integrations.accelerate_helpers import optimize_after_prepare

    accelerator = Accelerator()
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    model = optimize_after_prepare(model, input_shape=(1, 512))

That is — ``accelerator.prepare()`` runs first (handles device placement
and any distributed wrapping), then Agnitra runs on the prepared model.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from agnitra.integrations.huggingface import wrap_model


def _require_accelerate():
    try:
        import accelerate  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "agnitra.integrations.accelerate_helpers requires `accelerate`. "
            "Install it with `pip install accelerate`."
        ) from exc
    return accelerate


def optimize_after_prepare(
    model: "torch.nn.Module",
    *,
    input_shape: Optional[tuple] = None,
    input_tensor: Optional["torch.Tensor"] = None,
    agnitra_kwargs: Optional[Mapping[str, Any]] = None,
):
    """Run Agnitra on a model that has already been ``accelerator.prepare``-d.

    Either ``input_shape`` or ``input_tensor`` must be provided so the
    optimizer can profile a forward pass at the workload's actual shape.
    Any additional optimizer kwargs go in ``agnitra_kwargs``.
    """
    if input_shape is None and input_tensor is None and not (agnitra_kwargs and (
        "input_shape" in agnitra_kwargs or "input_tensor" in agnitra_kwargs
    )):
        raise ValueError(
            "optimize_after_prepare requires `input_shape=...` or "
            "`input_tensor=...` so the optimizer can profile a forward pass."
        )

    merged: dict = dict(agnitra_kwargs or {})
    if input_shape is not None:
        merged.setdefault("input_shape", input_shape)
    if input_tensor is not None:
        merged.setdefault("input_tensor", input_tensor)
    return wrap_model(model, agnitra_kwargs=merged)


__all__ = ["optimize_after_prepare"]
