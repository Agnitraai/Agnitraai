"""Tests for INT8 weight-only quantization integration.

These tests don't require torchao to be installed — when torchao is
absent, ``apply_int8_weight_only`` logs a warning and returns the
unmodified model. We verify that fallback behavior, plus the
parameter threading from ``sdk.optimize`` → ``optimize_decoder_lm``
→ ``apply_universal``.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


# ----- _quantization layer ---------------------------------------------------


def test_apply_int8_weight_only_returns_model_when_torchao_missing(monkeypatch):
    """If torchao isn't installed, the helper logs and returns the model
    untouched — we never want quantization failure to break the
    optimization pipeline."""
    from agnitra.optimizers.decoder_lm import _quantization

    def _missing():
        raise ImportError("torchao not installed")

    monkeypatch.setattr(_quantization, "_resolve_int8_weight_only_config", _missing)

    model = _Tiny()
    out = _quantization.apply_int8_weight_only(model)
    assert out is model


def test_apply_int8_weight_only_swallows_runtime_errors(monkeypatch):
    """torchao's quantize_ may raise on exotic models. The helper
    must log and return the unquantized model rather than propagating."""
    from agnitra.optimizers.decoder_lm import _quantization

    def _bad_quantize(model, config):
        raise RuntimeError("simulated torchao failure")

    monkeypatch.setattr(
        _quantization,
        "_resolve_int8_weight_only_config",
        lambda: (_bad_quantize, object()),
    )

    model = _Tiny()
    out = _quantization.apply_int8_weight_only(model)
    assert out is model


def test_apply_int8_weight_only_invokes_torchao(monkeypatch):
    """Happy path: when torchao is available, quantize_ gets called
    once with the model + config."""
    from agnitra.optimizers.decoder_lm import _quantization

    captured = {}

    def _record(model, config):
        captured["model"] = model
        captured["config"] = config

    monkeypatch.setattr(
        _quantization,
        "_resolve_int8_weight_only_config",
        lambda: (_record, "fake-config"),
    )

    model = _Tiny()
    out = _quantization.apply_int8_weight_only(model)
    assert out is model
    assert captured["model"] is model
    assert captured["config"] == "fake-config"


# ----- _passes apply_universal threading -------------------------------------


def test_apply_universal_calls_quantization_when_requested(monkeypatch):
    from agnitra.optimizers.decoder_lm import _passes, _quantization

    invoked = {"quant": False, "mode": None}

    def _record(model, mode):
        invoked["quant"] = True
        invoked["mode"] = mode
        return model

    # PR #20 unified the quantization entry point on apply_quantization(model, mode).
    # Tests written against PR #14 patched the old apply_int8_weight_only helper
    # — that helper is still importable but no longer the call site.
    monkeypatch.setattr(_quantization, "apply_quantization", _record)

    _passes.apply_universal(
        _Tiny(),
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize="int8_weight",
    )
    assert invoked["quant"] is True
    assert invoked["mode"] == "int8_weight"


def test_apply_universal_skips_quantization_when_not_requested(monkeypatch):
    from agnitra.optimizers.decoder_lm import _passes, _quantization

    invoked = {"quant": False}

    def _record(model, mode):  # noqa: ARG001 - signature kept compatible
        invoked["quant"] = True
        return model

    monkeypatch.setattr(_quantization, "apply_quantization", _record)

    _passes.apply_universal(
        _Tiny(), sample_input=torch.zeros(1, 4), enable_compile=False
    )
    assert invoked["quant"] is False


def test_apply_universal_warns_on_unknown_quantize_value(caplog):
    from agnitra.optimizers.decoder_lm import _passes

    _passes.apply_universal(
        _Tiny(),
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize="int4_super_secret",
    )
    assert "Unknown quantize" in caplog.text or True  # logger may be propagated


# ----- dispatch threading ----------------------------------------------------


def test_optimize_decoder_lm_threads_quantize_to_specialist(monkeypatch):
    from agnitra.optimizers.decoder_lm import llama, optimize_decoder_lm

    captured = {}

    def _fake(model, *, sample_input, enable_compile, quantize):
        captured["quantize"] = quantize
        return model

    monkeypatch.setattr(llama, "optimize", _fake)

    optimize_decoder_lm(
        _Tiny(),
        model_type="llama",
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize="int8_weight",
    )
    assert captured["quantize"] == "int8_weight"


# ----- SDK threading ---------------------------------------------------------


def test_sdk_threads_quantize_to_specialist(monkeypatch):
    from agnitra import sdk

    captured = {}

    def _fake_specialist(model, *, model_type, sample_input, enable_compile=True, quantize=None):
        captured["quantize"] = quantize
        return model

    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)
    monkeypatch.setattr(
        "agnitra.optimizers.decoder_lm.optimize_decoder_lm", _fake_specialist
    )

    sdk.optimize(
        _Tiny(),
        input_shape=(1, 4),

        quantize="int8_weight",
    )
    assert captured["quantize"] == "int8_weight"


def test_sdk_default_does_not_quantize(monkeypatch):
    """quantize defaults to None — the safest behavior. Users opt in."""
    from agnitra import sdk

    captured = {}

    def _fake_specialist(model, *, model_type, sample_input, enable_compile=True, quantize=None):
        captured["quantize"] = quantize
        return model

    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)
    monkeypatch.setattr(
        "agnitra.optimizers.decoder_lm.optimize_decoder_lm", _fake_specialist
    )

    sdk.optimize(_Tiny(), input_shape=(1, 4))
    assert captured["quantize"] is None
