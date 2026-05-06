"""Tests for INT4 / FP8 / auto quantization paths.

No torchao install required — we monkeypatch the resolvers and
torch.cuda.get_device_capability so the tests run on any host.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


# ----- apply_quantization dispatch -------------------------------------------


def test_apply_int4_path(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    captured = {}

    def _fake_apply(model, _config):
        captured["called"] = True

    monkeypatch.setattr(
        _quantization,
        "_resolve_int4_weight_only_config",
        lambda: (_fake_apply, "int4-cfg"),
    )
    out = _quantization.apply_quantization(_Tiny(), "int4_weight")
    assert isinstance(out, nn.Module)
    assert captured["called"] is True


def test_apply_fp8_path(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    captured = {}

    def _fake_apply(model, _config):
        captured["called"] = True

    monkeypatch.setattr(
        _quantization,
        "_resolve_fp8_weight_only_config",
        lambda: (_fake_apply, "fp8-cfg"),
    )
    out = _quantization.apply_quantization(_Tiny(), "fp8_weight")
    assert isinstance(out, nn.Module)
    assert captured["called"] is True


def test_unknown_mode_skips_silently(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    monkeypatch.setattr(
        _quantization,
        "_resolve_int8_weight_only_config",
        lambda: pytest.fail("INT8 path should not run for unknown mode"),
    )
    out = _quantization.apply_quantization(_Tiny(), "int999_imaginary")
    assert isinstance(out, nn.Module)


def test_torchao_missing_for_int4_falls_back(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    def _missing():
        raise ImportError("torchao>=0.5 required")

    monkeypatch.setattr(_quantization, "_resolve_int4_weight_only_config", _missing)
    out = _quantization.apply_quantization(_Tiny(), "int4_weight")
    # Returns model unchanged rather than raising.
    assert isinstance(out, nn.Module)


# ----- select_auto_mode ------------------------------------------------------


def test_auto_picks_fp8_on_h100(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (9, 0))
    assert _quantization.select_auto_mode() == "fp8_weight"


def test_auto_picks_int8_on_a100(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i=0: (8, 0))
    assert _quantization.select_auto_mode() == "int8_weight"


def test_auto_falls_back_to_int8_on_cpu(monkeypatch):
    from agnitra.optimizers.decoder_lm import _quantization

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _quantization.select_auto_mode() == "int8_weight"


def test_auto_resolves_to_concrete_mode(monkeypatch):
    """auto must dispatch to the underlying mode, not stay as 'auto'."""
    from agnitra.optimizers.decoder_lm import _quantization

    monkeypatch.setattr(_quantization, "select_auto_mode", lambda: "int4_weight")

    captured = {}

    def _fake_apply(model, _config):
        captured["called"] = True

    monkeypatch.setattr(
        _quantization,
        "_resolve_int4_weight_only_config",
        lambda: (_fake_apply, "cfg"),
    )

    _quantization.apply_quantization(_Tiny(), "auto")
    assert captured["called"] is True


# ----- back-compat ------------------------------------------------------------


def test_apply_int8_weight_only_alias_still_works(monkeypatch):
    """The 0.2.0 helper name remains importable."""
    from agnitra.optimizers.decoder_lm import _quantization

    captured = {}

    def _fake_apply(model, _config):
        captured["called"] = True

    monkeypatch.setattr(
        _quantization,
        "_resolve_int8_weight_only_config",
        lambda: (_fake_apply, "cfg"),
    )

    _quantization.apply_int8_weight_only(_Tiny())
    assert captured["called"] is True


# ----- _passes.apply_universal threading -------------------------------------


def test_apply_universal_threads_int4(monkeypatch):
    from agnitra.optimizers.decoder_lm import _passes, _quantization

    captured = {"mode": None}

    def _fake_apply(model, mode):
        captured["mode"] = mode
        return model

    monkeypatch.setattr(_quantization, "apply_quantization", _fake_apply)

    _passes.apply_universal(
        _Tiny(),
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize="int4_weight",
    )
    assert captured["mode"] == "int4_weight"


def test_apply_universal_threads_auto(monkeypatch):
    from agnitra.optimizers.decoder_lm import _passes, _quantization

    captured = {"mode": None}

    def _fake_apply(model, mode):
        captured["mode"] = mode
        return model

    monkeypatch.setattr(_quantization, "apply_quantization", _fake_apply)

    _passes.apply_universal(
        _Tiny(),
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize="auto",
    )
    assert captured["mode"] == "auto"


def test_apply_universal_skips_quantization_when_none(monkeypatch):
    from agnitra.optimizers.decoder_lm import _passes, _quantization

    invoked = {"called": False}

    def _fake_apply(model, mode):
        invoked["called"] = True
        return model

    monkeypatch.setattr(_quantization, "apply_quantization", _fake_apply)

    _passes.apply_universal(
        _Tiny(),
        sample_input=torch.zeros(1, 4),
        enable_compile=False,
        quantize=None,
    )
    assert invoked["called"] is False
