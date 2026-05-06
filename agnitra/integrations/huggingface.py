"""HuggingFace `transformers` integration for Agnitra AI.

The wedge promises: drop one line into your inference code and the model
gets faster. For HuggingFace users, that one line is::

    from agnitra.integrations.huggingface import AgnitraModel

    model = AgnitraModel.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        torch_dtype=torch.float16,
        agnitra_kwargs={"input_shape": (1, 512)},
    ).cuda()

Everything else — tokenizer use, ``.generate()`` calls, logits — stays
identical to vanilla ``transformers``. The wrapper loads a regular HF
model, runs ``agnitra.optimize(model)`` on it, and returns the
optimized ``nn.Module``. No metaclass tricks, no monkey patching.

Three entry points:

* :func:`wrap_model` — optimize a model you already loaded.
* :class:`AgnitraModel` — load via ``from_pretrained`` and optimize in
  one shot.
* :func:`optimize_pipeline` — replace the model inside a
  ``transformers.pipeline`` instance with the optimized version.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Type

from agnitra.core.runtime import RuntimeOptimizationResult
from agnitra.sdk import optimize as _agnitra_optimize


def _require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "agnitra.integrations.huggingface requires the `transformers` "
            "package. Install it with `pip install transformers`."
        ) from exc
    return transformers


# Default optimize() kwargs that suit inference workloads — RL search is
# expensive and adds variance, so the default is "off." Anything passed
# explicitly via ``agnitra_kwargs`` overrides these.
_INFERENCE_DEFAULTS: Mapping[str, Any] = {
    "enable_rl": False,
    "offline": True,
}


def wrap_model(
    model: "torch.nn.Module",
    *,
    agnitra_kwargs: Optional[Mapping[str, Any]] = None,
    return_result: bool = False,
):
    """Run ``agnitra.optimize`` on ``model`` and return the optimized module.

    The optimization metadata (baseline + optimized snapshots, usage
    event, optimizer notes) is attached to the returned module as
    ``model._agnitra_result`` for callers that care, and also returned
    explicitly when ``return_result=True``.
    """
    kwargs = dict(_INFERENCE_DEFAULTS)
    if agnitra_kwargs:
        kwargs.update(agnitra_kwargs)
    if "input_tensor" not in kwargs and "input_shape" not in kwargs:
        raise ValueError(
            "wrap_model requires `agnitra_kwargs={'input_shape': (...)}` "
            "or `agnitra_kwargs={'input_tensor': tensor}` so the optimizer "
            "can profile a representative forward pass."
        )

    result: RuntimeOptimizationResult = _agnitra_optimize(model, **kwargs)
    optimized = result.optimized_model
    try:
        # Attach metadata so callers can inspect baseline/optimized snapshots
        # without re-running the optimizer. Wrapped in try/except because
        # some module types reject attribute assignment.
        optimized._agnitra_result = result
    except Exception:
        pass
    if return_result:
        return optimized, result
    return optimized


class AgnitraModel:
    """``from_pretrained``-shaped wrapper that returns an optimized model.

    Mirrors ``transformers.AutoModelForCausalLM.from_pretrained()`` but
    runs ``agnitra.optimize`` before returning. Pass
    ``model_class=YourClass`` to use a different head (for example
    ``transformers.AutoModelForSeq2SeqLM``); defaults to
    ``AutoModelForCausalLM`` because that is the dominant inference
    workload Agnitra AI targets.

    Example::

        model = AgnitraModel.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.float16,
            agnitra_kwargs={"input_shape": (1, 512)},
        ).cuda()
        # Use `model` exactly like a transformers model.
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        model_class: Optional[Type[Any]] = None,
        agnitra_kwargs: Optional[Mapping[str, Any]] = None,
        **from_pretrained_kwargs: Any,
    ):
        transformers = _require_transformers()
        if model_class is None:
            model_class = transformers.AutoModelForCausalLM
        model = model_class.from_pretrained(
            pretrained_model_name_or_path, **from_pretrained_kwargs
        )
        return wrap_model(model, agnitra_kwargs=agnitra_kwargs)


def optimize_pipeline(
    pipe: Any,
    *,
    agnitra_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Replace the model inside a ``transformers.pipeline`` with the optimized version.

    The pipeline keeps its tokenizer, feature extractor, and other
    components untouched — only ``pipe.model`` is swapped.
    """
    if not hasattr(pipe, "model"):
        raise TypeError(
            "optimize_pipeline expected a transformers.pipeline-like object "
            "with a `.model` attribute."
        )
    optimized = wrap_model(pipe.model, agnitra_kwargs=agnitra_kwargs)
    pipe.model = optimized
    return pipe


__all__ = ["AgnitraModel", "wrap_model", "optimize_pipeline"]
