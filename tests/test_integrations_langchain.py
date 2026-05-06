"""Tests for the LangChain integration.

No real LangChain or transformers install required — we construct
fake LLM-shaped objects with the attributes the integration looks
for (`pipeline.model`) and monkeypatch the underlying optimizer.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


class _FakePipeline:
    def __init__(self, model):
        self.model = model


class _FakeHFPipelineLLM:
    """Mimics langchain_huggingface.HuggingFacePipeline shape."""

    def __init__(self, pipeline):
        self.pipeline = pipeline


def _patch_wrap_model(monkeypatch, captured):
    def _fake(model, *, agnitra_kwargs=None, return_result=False):
        captured["model"] = model
        captured["agnitra_kwargs"] = agnitra_kwargs
        return model  # identity replacement is enough for test

    monkeypatch.setattr("agnitra.integrations.langchain.wrap_model", _fake)


def test_optimize_llm_swaps_pipeline_model(monkeypatch):
    from agnitra.integrations.langchain import optimize_llm

    captured: dict = {}
    _patch_wrap_model(monkeypatch, captured)

    inner = _Tiny()
    pipeline = _FakePipeline(inner)
    llm = _FakeHFPipelineLLM(pipeline)

    out = optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 4)})
    assert out is llm
    assert captured["model"] is inner
    assert captured["agnitra_kwargs"] == {"input_shape": (1, 4)}
    # Pipeline.model is now the (identity-replaced) optimized model.
    assert pipeline.model is inner


def test_optimize_llm_handles_underscore_pipeline_attr(monkeypatch):
    """Some adapters expose `_pipeline` rather than `pipeline`."""
    from agnitra.integrations.langchain import optimize_llm

    captured: dict = {}
    _patch_wrap_model(monkeypatch, captured)

    class _AltLLM:
        def __init__(self, pipe):
            self._pipeline = pipe

    inner = _Tiny()
    llm = _AltLLM(_FakePipeline(inner))
    optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 4)})
    assert captured["model"] is inner


def test_optimize_llm_warns_and_returns_unchanged_for_unknown_type(monkeypatch):
    from agnitra.integrations.langchain import optimize_llm

    monkeypatch.setattr(
        "agnitra.integrations.langchain.wrap_model",
        lambda *a, **k: pytest.fail("wrap_model should not be called"),
    )

    class _Mystery:
        pass

    llm = _Mystery()
    out = optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 4)})
    assert out is llm  # unchanged


def test_optimize_llm_threads_quantize_kwarg(monkeypatch):
    from agnitra.integrations.langchain import optimize_llm

    captured: dict = {}
    _patch_wrap_model(monkeypatch, captured)

    pipeline = _FakePipeline(_Tiny())
    llm = _FakeHFPipelineLLM(pipeline)
    optimize_llm(
        llm,
        agnitra_kwargs={"input_shape": (1, 4), "quantize": "int8_weight"},
    )
    assert captured["agnitra_kwargs"]["quantize"] == "int8_weight"
