from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agnitra.core.billing import BenchmarkSample, compute_cost_delta, summarise_benchmark
from agnitra.core.licensing import LicenseManager, LicenseValidationError
from agnitra.core.runtime import suggest_kernel_config


def _license_payload(expires_in_days: int = 30) -> dict:
    return {
        "license_key": "LIC-123",
        "customer": "ExampleCorp",
        "expires_at": (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat(),
        "seats": 2,
        "features": ["offline", "per_gpu"],
        "metadata": {"tier": "enterprise"},
    }


def test_license_manager_loads_and_enforces(tmp_path):
    license_file = tmp_path / "license.json"
    manager = LicenseManager(secret="shared-secret", storage_path=license_file)
    manager.save(_license_payload())

    record = manager.load()
    assert record.customer == "ExampleCorp"

    manager.ensure_feature("OFFLINE")

    manager.checkout_seat("seat-1")
    manager.checkout_seat("seat-2")
    with pytest.raises(LicenseValidationError):
        manager.checkout_seat("seat-3")
    manager.release_seat("seat-2")

    usage = manager.register_gpu_run(org_id="org-demo")
    assert usage["active_gpu_count"] >= 0
    assert usage["license_key"] == "LIC-123"


def test_license_manager_rejects_missing_file(tmp_path):
    manager = LicenseManager(storage_path=tmp_path / "missing.json")
    with pytest.raises(LicenseValidationError):
        manager.load()


def test_compute_cost_delta_handles_benchmark():
    benchmark = summarise_benchmark(
        {
            "baseline": {"latency_ms": 120.0, "tokens_per_sec": 90.0},
            "optimized": {"latency_ms": 80.0, "tokens_per_sec": 135.0},
            "tokens_processed": 2048,
        }
    )
    assert isinstance(benchmark, BenchmarkSample)

    result = compute_cost_delta(benchmark, cost_per_second_gpu=3.0)
    assert result.cost_before > result.cost_after
    assert result.cost_saving > 0
    assert result.tokens_per_sec_uplift_pct > 0


def test_suggest_kernel_config_reflects_target():
    telemetry = {"latency_ms": 10.5, "memory_bytes": 512 * 1024}
    config_gpu = suggest_kernel_config("GPU-A100", telemetry)
    config_cpu = suggest_kernel_config("CPU-x86", telemetry)

    assert config_gpu["block_size"] != config_cpu["block_size"]
    assert config_gpu["latency_budget_ms"] == telemetry["latency_ms"]
