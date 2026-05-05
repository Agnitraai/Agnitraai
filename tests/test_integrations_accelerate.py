"""Tests for agnitra.integrations.accelerate_helpers.

The helper is a thin facade over ``wrap_model`` so the tests just verify
the input-spec validation and kwargs threading. No GPU, no
``accelerate`` install required.
"""
from __future__ import annotations

import pytest
import torch
from torch import nn


class _Tiny(nn.Module):
    def forward(self, x):
        return x + 1


def _fake_optimize_factory(monkeypatch, captured):
    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )

    def _fake(model, **kwargs):
        captured["called_with"] = kwargs
        snap = OptimizationSnapshot(
            latency_ms=10.0, tokens_per_sec=100.0, tokens_processed=10,
            gpu_utilization=None, telemetry={}, metadata={},
        )
        return RuntimeOptimizationResult(
            optimized_model=model, baseline=snap, optimized=snap,
            usage_event=None, notes={},
        )

    monkeypatch.setattr("agnitra.integrations.huggingface._agnitra_optimize", _fake)


def test_optimize_after_prepare_input_shape(monkeypatch):
    from agnitra.integrations.accelerate_helpers import optimize_after_prepare

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    model = _Tiny()
    out = optimize_after_prepare(model, input_shape=(1, 4))
    assert out is model
    assert captured["called_with"]["input_shape"] == (1, 4)


def test_optimize_after_prepare_input_tensor(monkeypatch):
    from agnitra.integrations.accelerate_helpers import optimize_after_prepare

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    tensor = torch.zeros(1, 4)
    optimize_after_prepare(_Tiny(), input_tensor=tensor)
    assert captured["called_with"]["input_tensor"] is tensor


def test_optimize_after_prepare_requires_input_spec(monkeypatch):
    from agnitra.integrations.accelerate_helpers import optimize_after_prepare

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    with pytest.raises(ValueError, match="input_shape"):
        optimize_after_prepare(_Tiny())


def test_optimize_after_prepare_extra_kwargs_pass_through(monkeypatch):
    from agnitra.integrations.accelerate_helpers import optimize_after_prepare

    captured: dict = {}
    _fake_optimize_factory(monkeypatch, captured)

    optimize_after_prepare(
        _Tiny(),
        input_shape=(1, 4),
        agnitra_kwargs={"enable_rl": True, "project_id": "myproj"},
    )
    assert captured["called_with"]["enable_rl"] is True
    assert captured["called_with"]["project_id"] == "myproj"
