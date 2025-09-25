"""Tests for the simulated PPO-based RL optimizer."""

from __future__ import annotations

import pytest

from agnitra.core.optimizer import (
    PPOKernelOptimizerConfig,
    PPOKernelOptimizer,
    run_dummy_training_loop,
    summarize_kernel_telemetry,
)

try:  # NumPy is an optional dependency; skip when unavailable.
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dep missing
    np = None  # type: ignore


pytestmark = pytest.mark.skipif(np is None, reason="NumPy is required for RL optimizer tests")


def test_summarize_kernel_telemetry_uses_defaults_when_empty() -> None:
    stats = summarize_kernel_telemetry(telemetry=None)
    assert stats.baseline_latency_ms > 0
    assert stats.baseline_tokens_per_sec > 0
    assert stats.event_count == 0


def test_dummy_training_loop_returns_result_without_sb3() -> None:
    result = run_dummy_training_loop(total_timesteps=64, prefer_sb3=False, seed=11)
    assert result.metadata["strategy"] == "random-search"
    assert result.metadata["dependency_fallback"] is True
    assert result.tile_size in {16, 32, 64, 128}
    assert result.unroll_factor in {1, 2, 4, 8}


def test_optimizer_train_respects_prefer_sb3_flag() -> None:
    telemetry = [
        {"cuda_time_ms": 5.5, "tokens_per_sec": 1500},
        {"cuda_time_ms": 7.5, "tokens_per_sec": 1100},
    ]
    summary = summarize_kernel_telemetry(telemetry)
    config = PPOKernelOptimizerConfig(
        telemetry_summary=summary,
        prefer_sb3=False,
        seed=21,
        total_timesteps=128,
    )
    optimizer = PPOKernelOptimizer(config=config)
    result = optimizer.train()
    assert result.metadata["dependency_fallback"] is True
    assert result.metadata.get("dependency_fallback_reason") == "disabled"
