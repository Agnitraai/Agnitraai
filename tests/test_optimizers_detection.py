"""Tests for agnitra.optimizers detection + routing.

These tests stay GPU-free and transformers-free. We construct tiny
``nn.Module`` instances with synthetic ``config`` attributes to exercise
the detection paths, and we monkeypatch the underlying optimizer to
verify the SDK passes through unchanged for unsupported architectures.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from agnitra.optimizers import (
    SUPPORTED_DECODER_LM_TYPES,
    detect_architecture,
    is_supported,
)
from agnitra.optimizers.detection import _looks_like_decoder_lm


class _BareModule(nn.Module):
    """No config, no decoder structure — should be undetectable."""

    def forward(self, x):
        return x


class _StructuralDecoderLM(nn.Module):
    """Has the structural fingerprint but no config.model_type."""

    def __init__(self):
        super().__init__()

        # Mimic transformers' shape: an `embed_tokens` and an attention.
        self.embed_tokens = nn.Embedding(100, 16)
        self.self_attention = _DummyAttention()
        self.lm_head = nn.Linear(16, 100)

    def forward(self, x):
        return self.lm_head(self.self_attention(self.embed_tokens(x)))


class _DummyAttention(nn.Module):
    """Class name contains 'Attention' so the structural test fires."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16)

    def forward(self, x):
        return self.linear(x)


# ----- registry -----------------------------------------------------------


def test_supported_set_contains_llama_mistral_qwen():
    """Sanity check that the canonical wedge architectures are in the set."""
    for arch in ("llama", "mistral", "qwen2", "gemma", "phi3"):
        assert arch in SUPPORTED_DECODER_LM_TYPES


def test_is_supported_is_case_sensitive():
    assert is_supported("llama") is True
    # detect_architecture lowercases; is_supported does not. That keeps
    # the registry the single source of truth and prevents drive-by
    # additions via casing.
    assert is_supported("LLAMA") is False


def test_is_supported_handles_none_and_empty():
    assert is_supported(None) is False
    assert is_supported("") is False


def test_is_supported_rejects_unknown():
    assert is_supported("bert") is False
    assert is_supported("vit") is False
    assert is_supported("decoder_lm_generic") is False  # structural fallback


# ----- detection ----------------------------------------------------------


def test_detect_returns_none_for_bare_module():
    assert detect_architecture(_BareModule()) is None


def test_detect_uses_config_model_type_when_present():
    model = _BareModule()
    model.config = types.SimpleNamespace(model_type="LLaMa")  # arbitrary case
    assert detect_architecture(model) == "llama"


def test_detect_falls_back_to_structural_pattern():
    # No config, but has embed_tokens + something with "Attention" in class name.
    model = _StructuralDecoderLM()
    assert detect_architecture(model) == "decoder_lm_generic"


def test_structural_check_directly():
    assert _looks_like_decoder_lm(_StructuralDecoderLM()) is True
    assert _looks_like_decoder_lm(_BareModule()) is False


def test_detect_handles_attribute_errors_gracefully():
    """A model whose `config` access raises shouldn't crash detection."""

    class _BadModel(nn.Module):
        @property
        def config(self):
            raise RuntimeError("config explodes")

        def forward(self, x):
            return x

    # Detection should silently fall back to structural test (also returns
    # False here) and produce None.
    assert detect_architecture(_BadModel()) is None


# ----- SDK pass-through -----------------------------------------------------


def test_optimize_passes_through_unsupported_architecture(monkeypatch):
    """Unsupported architectures must NOT call the underlying optimizer."""
    from agnitra import sdk

    called = {"optimize": False}

    def _fail(*_args, **_kwargs):
        called["optimize"] = True
        raise AssertionError("optimizer should not run for unsupported arch")

    # Patch every optimizer entry point we might accidentally fall into.
    monkeypatch.setattr("agnitra.sdk._optimize_model", _fail)
    monkeypatch.setattr(
        "agnitra.core.runtime.RuntimeOptimizationAgent.optimize",
        _fail,
        raising=False,
    )

    model = _BareModule()
    result = sdk.optimize(model, input_shape=(1, 4))

    assert called["optimize"] is False
    assert result.optimized_model is model
    assert result.notes["passthrough"] is True
    assert result.notes["reason"] == "unsupported_architecture"
    assert result.notes["detected_architecture"] is None


def test_optimize_passes_through_explicitly_unsupported_arch(monkeypatch):
    """A model with config.model_type='bert' must also pass through."""
    from agnitra import sdk

    monkeypatch.setattr(
        "agnitra.sdk._optimize_model",
        lambda *a, **k: pytest.fail("optimizer should not run"),
    )

    model = _BareModule()
    model.config = types.SimpleNamespace(model_type="bert")
    result = sdk.optimize(model, input_shape=(1, 4))

    assert result.optimized_model is model
    assert result.notes["detected_architecture"] == "bert"
