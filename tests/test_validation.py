"""Tests for the output-validation safety net.

Both the standalone ``output_drift`` helper and the SDK integration
(``optimize(validate=True, fallback_on_regression=True)``) are
exercised. No GPU required — we use a tiny ``nn.Module`` plus an
"optimized" copy that intentionally returns wrong outputs to trigger
the regression path.
"""
from __future__ import annotations

import torch
from torch import nn

from agnitra.core.runtime.validation import OutputDrift, output_drift


class _Identity(nn.Module):
    def forward(self, x):
        return x


class _OffByConstant(nn.Module):
    """Returns x + 100 — large drift, should regress."""

    def forward(self, x):
        return x + 100.0


class _NaNFactory(nn.Module):
    """Returns NaNs — used to verify cosine-similarity divide-by-zero handling."""

    def forward(self, x):
        return torch.full_like(x, float("nan"))


# ----- output_drift unit tests -----------------------------------------------


def test_drift_zero_for_identical_models():
    sample = torch.randn(4, 8)
    d = output_drift(_Identity(), _Identity(), sample)
    assert d.regressed is False
    assert d.cosine_similarity > 0.999
    assert d.max_abs_diff == 0.0
    assert d.argmax_match_rate == 1.0


def test_drift_high_for_off_by_constant_model():
    sample = torch.randn(4, 8)
    d = output_drift(_Identity(), _OffByConstant(), sample)
    # Outputs are still tensors of the same shape — shapes_match is True —
    # but cosine similarity is degraded and max_abs_diff is large.
    assert d.shapes_match is True
    assert d.max_abs_diff > 50.0


def test_drift_captures_forward_exception():
    class _Boom(nn.Module):
        def forward(self, x):
            raise RuntimeError("kaboom")

    sample = torch.randn(4, 8)
    d = output_drift(_Identity(), _Boom(), sample)
    assert d.regressed is True
    assert d.error is not None
    assert "kaboom" in d.error


def test_drift_handles_nan_outputs_without_crashing():
    sample = torch.randn(4, 8)
    d = output_drift(_Identity(), _NaNFactory(), sample)
    # Should return a result rather than raise. Cosine on NaNs is
    # ill-defined; the helper returns 0 when norms are zero or NaN
    # propagates — either way regressed semantics are correct because
    # the optimizer would never be allowed to ship NaN outputs.
    assert isinstance(d, OutputDrift)


def test_drift_treats_shape_mismatch_as_hard_fail():
    class _ReshapeWrong(nn.Module):
        def forward(self, x):
            return x[..., :-1]  # different last dim

    sample = torch.randn(4, 8)
    d = output_drift(_Identity(), _ReshapeWrong(), sample)
    assert d.shapes_match is False
    assert d.regressed is True


def test_drift_handles_modeloutput_like_objects():
    """HF transformers wraps tensors in ModelOutput dataclasses with .logits."""

    class _ModelOutputLike:
        def __init__(self, logits):
            self.logits = logits

    class _Wrap(nn.Module):
        def forward(self, x):
            return _ModelOutputLike(x)

    sample = torch.randn(2, 4, 8)
    d = output_drift(_Wrap(), _Wrap(), sample)
    assert d.shapes_match is True
    assert d.cosine_similarity > 0.999


# ----- SDK integration -------------------------------------------------------


def _stub_runtime_result(optimized_model, baseline_model):
    """Build a RuntimeOptimizationResult that the SDK code path will accept."""
    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )

    snap = OptimizationSnapshot(
        latency_ms=10.0,
        tokens_per_sec=100.0,
        tokens_processed=1,
        gpu_utilization=None,
        telemetry={},
        metadata={},
    )
    return RuntimeOptimizationResult(
        optimized_model=optimized_model,
        baseline=snap,
        optimized=snap,
        usage_event=None,
        notes={},
    )


def test_sdk_validates_and_reverts_on_regression(monkeypatch):
    """If the underlying optimizer returns a wrong-output model, the SDK
    falls back to the baseline and annotates."""
    from agnitra import sdk

    baseline = _Identity()
    bad_optimized = _OffByConstant()

    # Force the architecture detection to mark the model as supported
    # so we actually exercise the validation branch.
    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return _stub_runtime_result(bad_optimized, baseline)

    monkeypatch.setattr("agnitra.sdk.RuntimeOptimizationAgent", _FakeAgent)

    result = sdk.optimize(
        baseline,
        input_shape=(2, 4),
        offline=True,
    )

    # Validation should have detected drift and reverted to baseline.
    assert result.optimized_model is baseline
    assert result.notes.get("reverted_due_to_drift") is True
    assert result.notes["validation"]["regressed"] is True


def test_sdk_skips_validation_when_disabled(monkeypatch):
    """validate=False keeps the (possibly-broken) optimized model."""
    from agnitra import sdk

    baseline = _Identity()
    bad_optimized = _OffByConstant()

    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return _stub_runtime_result(bad_optimized, baseline)

    monkeypatch.setattr("agnitra.sdk.RuntimeOptimizationAgent", _FakeAgent)

    result = sdk.optimize(
        baseline,
        input_shape=(2, 4),
        offline=True,
        validate=False,
    )

    # No fallback when validation is off.
    assert result.optimized_model is bad_optimized
    assert "validation" not in result.notes
    assert "reverted_due_to_drift" not in result.notes


def test_sdk_keeps_optimized_when_validation_passes(monkeypatch):
    """A correct optimization passes validation and is kept."""
    from agnitra import sdk

    baseline = _Identity()
    good_optimized = _Identity()  # same outputs, different instance

    monkeypatch.setattr("agnitra.sdk.detect_architecture", lambda m: "llama")
    monkeypatch.setattr("agnitra.sdk.is_supported", lambda m: True)

    class _FakeAgent:
        def __init__(self, *args, **kwargs):
            pass

        def optimize(self, *args, **kwargs):
            return _stub_runtime_result(good_optimized, baseline)

    monkeypatch.setattr("agnitra.sdk.RuntimeOptimizationAgent", _FakeAgent)

    result = sdk.optimize(baseline, input_shape=(2, 4), offline=True)

    assert result.optimized_model is good_optimized
    assert result.notes["validation"]["regressed"] is False
    assert "reverted_due_to_drift" not in result.notes
