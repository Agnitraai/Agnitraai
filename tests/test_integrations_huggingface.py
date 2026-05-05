"""Tests for agnitra.integrations.huggingface.

These tests do NOT require ``transformers`` or a GPU. We monkeypatch the
underlying ``agnitra.optimize`` (the public SDK call our wrappers
delegate to) and provide a tiny ``transformers``-shaped stub for the
``AgnitraModel.from_pretrained`` test. This keeps integration tests
fast and runnable in CI without LLM weights.
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


def _fake_optimize_factory(monkeypatch, captured):
    """Replace agnitra.sdk.optimize with a recorder that returns a result."""
    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )

    def _fake(model, **kwargs):
        captured["called_with"] = kwargs
        captured["model"] = model
        snap = OptimizationSnapshot(
            latency_ms=10.0,
            tokens_per_sec=100.0,
            tokens_processed=10,
            gpu_utilization=None,
            telemetry={},
            metadata={},
        )
        return RuntimeOptimizationResult(
            optimized_model=model,
            baseline=snap,
            optimized=snap,
            usage_event=None,
            notes={"fake": True},
        )

    monkeypatch.setattr("agnitra.integrations.huggingface._agnitra_optimize", _fake)


def test_wrap_model_passes_through_kwargs(monkeypatch):
    from agnitra.integrations.huggingface import wrap_model

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    model = _Tiny()
    optimized = wrap_model(model, agnitra_kwargs={"input_shape": (1, 4)})

    assert optimized is model
    assert captured["called_with"]["input_shape"] == (1, 4)
    # Defaults set for inference workloads.
    assert captured["called_with"]["enable_rl"] is False
    assert captured["called_with"]["offline"] is True
    # Result is attached to the optimized module for callers that want it.
    assert getattr(optimized, "_agnitra_result").notes == {"fake": True}


def test_wrap_model_explicit_kwargs_override_defaults(monkeypatch):
    from agnitra.integrations.huggingface import wrap_model

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    wrap_model(
        _Tiny(),
        agnitra_kwargs={"input_shape": (1, 4), "enable_rl": True, "offline": False},
    )
    assert captured["called_with"]["enable_rl"] is True
    assert captured["called_with"]["offline"] is False


def test_wrap_model_requires_input_spec(monkeypatch):
    from agnitra.integrations.huggingface import wrap_model

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    with pytest.raises(ValueError, match="input_shape"):
        wrap_model(_Tiny())


def test_wrap_model_returns_result_when_requested(monkeypatch):
    from agnitra.integrations.huggingface import wrap_model

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    optimized, result = wrap_model(
        _Tiny(),
        agnitra_kwargs={"input_shape": (1, 4)},
        return_result=True,
    )
    assert optimized is result.optimized_model
    assert result.notes == {"fake": True}


def test_optimize_pipeline_replaces_model(monkeypatch):
    from agnitra.integrations.huggingface import optimize_pipeline

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    class _FakePipeline:
        def __init__(self, model):
            self.model = model

    pipe = _FakePipeline(_Tiny())
    out = optimize_pipeline(pipe, agnitra_kwargs={"input_shape": (1, 4)})
    assert out is pipe
    assert pipe.model is captured["model"]


def test_optimize_pipeline_rejects_objects_without_model():
    from agnitra.integrations.huggingface import optimize_pipeline

    with pytest.raises(TypeError, match=".model"):
        optimize_pipeline(object(), agnitra_kwargs={"input_shape": (1, 4)})


def test_agnitra_model_from_pretrained_uses_stubbed_transformers(monkeypatch):
    """AgnitraModel calls model_class.from_pretrained, then wraps the result."""
    from agnitra.integrations import huggingface as hf

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    inner = _Tiny()

    class _FakeModelClass:
        called: dict = {}

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            cls.called["model_id"] = model_id
            cls.called["kwargs"] = kwargs
            return inner

    # Inject a stub `transformers` module so AgnitraModel.from_pretrained
    # accepts the import — but pass model_class explicitly so the test
    # doesn't depend on AutoModelForCausalLM existing on the stub.
    fake_tx = types.ModuleType("transformers")
    monkeypatch.setitem(sys.modules, "transformers", fake_tx)

    out = hf.AgnitraModel.from_pretrained(
        "fake/model",
        model_class=_FakeModelClass,
        torch_dtype="float16",
        agnitra_kwargs={"input_shape": (1, 4)},
    )
    assert out is inner
    assert _FakeModelClass.called["model_id"] == "fake/model"
    assert _FakeModelClass.called["kwargs"] == {"torch_dtype": "float16"}


def test_agnitra_model_raises_clear_error_without_transformers(monkeypatch):
    """If transformers isn't installed, the error message points at the fix."""
    from agnitra.integrations import huggingface as hf

    # Force the import to fail.
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="pip install transformers"):
        hf.AgnitraModel.from_pretrained("fake/model")
