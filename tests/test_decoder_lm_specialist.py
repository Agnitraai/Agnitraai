"""Tests for the decoder-LM specialist optimizers.

Two layers under test:

1. The pass functions in agnitra/optimizers/decoder_lm/_passes.py —
   verified to be no-throw on minimal inputs and to set the right
   config attributes when present.

2. The dispatch from sdk.optimize through optimize_decoder_lm down to
   each architecture handler — verified by monkeypatching the pass
   sequence and asserting it ran.

No GPU, no transformers — uses plain torch.nn.Module with synthetic
``config`` attributes.
"""
from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from agnitra.optimizers.decoder_lm import _passes, optimize_decoder_lm


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


def _model_with_config(model_type: str = "llama"):
    m = _Tiny()
    m.config = types.SimpleNamespace(model_type=model_type)
    return m


# ----- pass-level unit tests -------------------------------------------------


def test_enable_tf32_does_not_crash():
    # Just verify the call is safe — actual flag setting is global state
    # which we don't want to assert against.
    _passes.enable_tf32()


def test_ensure_sdpa_sets_attn_implementation_when_config_present():
    model = _model_with_config()
    _passes.ensure_sdpa_attention(model)
    assert getattr(model.config, "_attn_implementation", None) == "sdpa"


def test_ensure_sdpa_is_safe_without_config():
    model = _Tiny()
    out = _passes.ensure_sdpa_attention(model)
    assert out is model  # returns same instance


def test_enable_static_kv_cache_sets_flag():
    model = _model_with_config()
    _passes.enable_static_kv_cache(model)
    assert model.config.cache_implementation == "static"


def test_enable_static_kv_cache_safe_without_config():
    model = _Tiny()
    out = _passes.enable_static_kv_cache(model)
    assert out is model


def test_compile_for_decode_returns_callable_module(monkeypatch):
    """torch.compile may legitimately fail on tiny test modules; the
    helper catches that and returns the original model. We don't want
    to actually invoke the compiler in unit tests (slow), so we patch
    torch.compile to a recorder."""
    called = {}

    def _fake_compile(model, *, mode, fullgraph):
        called["mode"] = mode
        called["fullgraph"] = fullgraph
        return model

    monkeypatch.setattr("torch.compile", _fake_compile)
    model = _Tiny()
    out = _passes.compile_for_decode(model)
    assert out is model
    assert called["mode"] == "reduce-overhead"
    assert called["fullgraph"] is False


def test_apply_universal_runs_each_pass(monkeypatch):
    invoked = []

    def _record(name):
        def _fn(*args, **kwargs):
            invoked.append(name)
            return args[0] if args else None

        return _fn

    monkeypatch.setattr(_passes, "enable_tf32", _record("tf32"))
    monkeypatch.setattr(_passes, "ensure_sdpa_attention", _record("sdpa"))
    monkeypatch.setattr(_passes, "enable_static_kv_cache", _record("kv_cache"))
    monkeypatch.setattr(_passes, "compile_for_decode", _record("compile"))

    model = _Tiny()
    out = _passes.apply_universal(model, sample_input=torch.zeros(1, 4))

    assert invoked == ["tf32", "sdpa", "kv_cache", "compile"]
    assert out is model


def test_apply_universal_skip_compile_for_tests():
    """enable_compile=False keeps the test fast even without monkeypatch."""
    invoked = []

    def _record(_model, *args, **kwargs):
        invoked.append("compile")
        return _model

    # Default path (no monkeypatch) but enable_compile=False:
    out = _passes.apply_universal(_Tiny(), sample_input=torch.zeros(1, 4), enable_compile=False)
    assert isinstance(out, nn.Module)


# ----- dispatch -------------------------------------------------------------


def test_optimize_decoder_lm_dispatches_llama(monkeypatch):
    """model_type='llama' must reach the Llama specialist module."""
    from agnitra.optimizers.decoder_lm import llama

    called = {}

    def _fake(model, *, sample_input, enable_compile, **_kw):
        called["arch"] = "llama"
        return model

    monkeypatch.setattr(llama, "optimize", _fake)

    model = _Tiny()
    optimize_decoder_lm(
        model,
        model_type="llama",
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
    )
    assert called["arch"] == "llama"


@pytest.mark.parametrize(
    "model_type,expected_module",
    [
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("qwen2", "qwen2"),
        ("qwen2_moe", "qwen2"),
        ("gemma", "gemma"),
        ("gemma2", "gemma"),
    ],
)
def test_optimize_decoder_lm_dispatches_by_model_type(monkeypatch, model_type, expected_module):
    from agnitra.optimizers.decoder_lm import gemma, mistral, qwen2

    target = {"mistral": mistral, "qwen2": qwen2, "gemma": gemma}[expected_module]
    called = {"hit": False}

    def _fake(model, *, sample_input, enable_compile, **_kw):
        called["hit"] = True
        return model

    monkeypatch.setattr(target, "optimize", _fake)
    optimize_decoder_lm(
        _Tiny(),
        model_type=model_type,
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
    )
    assert called["hit"], f"expected {expected_module}.optimize to be invoked for {model_type}"


def test_optimize_decoder_lm_falls_back_to_generic_for_unknown_in_set(monkeypatch):
    """An architecture in the registry but without a tuned specialist
    falls through to the generic decoder-LM passes."""
    invoked = {"universal": False}

    def _fake_universal(model, *, sample_input, enable_compile, **_kw):
        invoked["universal"] = True
        return model

    monkeypatch.setattr(_passes, "apply_universal", _fake_universal)

    out = optimize_decoder_lm(
        _Tiny(),
        model_type="phi3",  # in the registry, no specialist module yet
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
    )
    assert invoked["universal"] is True
    assert isinstance(out, nn.Module)


# ----- SDK integration ------------------------------------------------------


def test_sdk_routes_supported_arch_to_specialist(monkeypatch):
    """sdk.optimize on a Llama model must route through the specialist
    rather than the legacy agent.optimize path."""
    from agnitra import sdk

    called = {"specialist": False, "agent": False}

    def _fake_specialist(model, *, model_type, sample_input, enable_compile=True, **_kw):
        called["specialist"] = True
        called["model_type"] = model_type
        return model

    class _FailingAgent:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            called["agent"] = True
            raise AssertionError("legacy agent must not run for ring-1 architectures")

    # Force detection to "llama" so the specialist branch fires.
    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)
    monkeypatch.setattr(
        "agnitra.optimizers.decoder_lm.optimize_decoder_lm", _fake_specialist
    )
    monkeypatch.setattr("agnitra.sdk.RuntimeOptimizationAgent", _FailingAgent)

    result = sdk.optimize(_Tiny(), input_shape=(1, 4))
    assert called["specialist"] is True
    assert called["agent"] is False
    assert called["model_type"] == "llama"
    assert result.notes["specialist"] is True
    assert result.notes["detected_architecture"] == "llama"


def test_sdk_use_specialist_false_falls_back_to_agent(monkeypatch):
    """use_specialist=False forces the legacy agent.optimize path even
    for ring-1 architectures."""
    from agnitra import sdk

    called = {"specialist": False, "agent": False}

    def _fake_specialist(*args, **kwargs):
        called["specialist"] = True
        raise AssertionError("specialist must not run when use_specialist=False")

    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, model, *args, **kwargs):
            called["agent"] = True
            snap = OptimizationSnapshot(
                latency_ms=10.0, tokens_per_sec=100.0, tokens_processed=1,
                gpu_utilization=None, telemetry={}, metadata={},
            )
            return RuntimeOptimizationResult(
                optimized_model=model, baseline=snap, optimized=snap,
                usage_event=None, notes={},
            )

    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)
    monkeypatch.setattr(
        "agnitra.optimizers.decoder_lm.optimize_decoder_lm", _fake_specialist
    )
    monkeypatch.setattr("agnitra.sdk.RuntimeOptimizationAgent", _FakeAgent)

    sdk.optimize(_Tiny(), input_shape=(1, 4), use_specialist=False)
    assert called["agent"] is True
    assert called["specialist"] is False
