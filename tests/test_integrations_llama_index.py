"""Tests for the LlamaIndex integration."""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


class _FakeLlamaIndexLLM:
    """Mimics llama_index.llms.huggingface.HuggingFaceLLM shape."""

    def __init__(self, model):
        self._model = model


def _patch_wrap_model(monkeypatch, captured):
    def _fake(model, *, agnitra_kwargs=None, return_result=False):
        captured["model"] = model
        captured["agnitra_kwargs"] = agnitra_kwargs
        # Return a different instance to verify the LLM swap actually happens.
        replacement = _Tiny()
        captured["replacement"] = replacement
        return replacement

    monkeypatch.setattr("agnitra.integrations.llama_index.wrap_model", _fake)


def test_optimize_llm_swaps_inner_model(monkeypatch):
    from agnitra.integrations.llama_index import optimize_llm

    captured: dict = {}
    _patch_wrap_model(monkeypatch, captured)

    inner = _Tiny()
    llm = _FakeLlamaIndexLLM(inner)

    out = optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 4)})
    assert out is llm
    assert captured["model"] is inner
    # The LLM's _model attribute now points to the replacement.
    assert llm._model is captured["replacement"]


def test_optimize_llm_handles_model_attr_variant(monkeypatch):
    from agnitra.integrations.llama_index import optimize_llm

    captured: dict = {}
    _patch_wrap_model(monkeypatch, captured)

    class _AltLLM:
        def __init__(self, m):
            self.model = m

    inner = _Tiny()
    llm = _AltLLM(inner)
    optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 4)})
    assert captured["model"] is inner
    assert llm.model is captured["replacement"]


def test_optimize_llm_warns_and_returns_unchanged_for_unknown_type(monkeypatch):
    from agnitra.integrations.llama_index import optimize_llm

    monkeypatch.setattr(
        "agnitra.integrations.llama_index.wrap_model",
        lambda *a, **k: pytest.fail("wrap_model should not be called"),
    )

    class _Mystery:
        pass

    out = optimize_llm(_Mystery(), agnitra_kwargs={"input_shape": (1, 4)})
    assert isinstance(out, _Mystery)


def test_optimize_llm_skips_non_module_attributes(monkeypatch):
    """Don't mistake unrelated `model` attributes (e.g. a string) for a module."""
    from agnitra.integrations.llama_index import optimize_llm

    monkeypatch.setattr(
        "agnitra.integrations.llama_index.wrap_model",
        lambda *a, **k: pytest.fail("wrap_model should not be called"),
    )

    class _StringModel:
        def __init__(self):
            self.model = "meta-llama/Meta-Llama-3-8B"

    out = optimize_llm(_StringModel(), agnitra_kwargs={"input_shape": (1, 4)})
    assert out.model == "meta-llama/Meta-Llama-3-8B"
