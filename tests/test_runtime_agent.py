import pytest
import torch

from agnitra.core.metering import UsageMeter
from agnitra.core.runtime.agent import RuntimeOptimizationAgent


class _Snapshot:
    def __init__(self, latency_ms, tokens_per_sec, tokens_processed, gpu_util=None):
        self.latency_ms = latency_ms
        self.tokens_per_sec = tokens_per_sec
        self.tokens_processed = tokens_processed
        self.gpu_utilization = gpu_util


def test_usage_meter_computes_billing_figures():
    meter = UsageMeter(rate_per_gpu_hour=4.0, margin_pct=0.1, currency="USD")
    baseline = _Snapshot(latency_ms=500.0, tokens_per_sec=2048.0, tokens_processed=1024, gpu_util=55.0)
    optimized = _Snapshot(latency_ms=250.0, tokens_per_sec=4096.0, tokens_processed=1024, gpu_util=60.0)

    event = meter.record_optimization(
        project_id="proj-123",
        model_name="demo-model",
        baseline_snapshot=baseline,
        optimized_snapshot=optimized,
        metadata={"stage": "unit-test"},
    )

    assert event.gpu_hours_before == pytest.approx((1024 / 2048.0) / 3600.0, rel=1e-6)
    assert event.gpu_hours_after == pytest.approx((1024 / 4096.0) / 3600.0, rel=1e-6)
    assert pytest.approx(event.performance_uplift_pct, rel=1e-3) == 100.0
    assert event.tokens_processed == 1024
    assert event.currency == "USD"
    assert event.metadata["stage"] == "unit-test"
    assert event.gpu_util_before == pytest.approx(55.0)
    assert event.gpu_util_after == pytest.approx(60.0)
    assert event.cost_before > event.cost_after
    assert event.cost_savings > 0.0
    assert event.total_billable == pytest.approx(event.usage_charge + event.success_fee, rel=1e-6)


def test_runtime_agent_generates_usage_event(monkeypatch):
    agent = RuntimeOptimizationAgent(
        usage_meter=UsageMeter(rate_per_gpu_hour=3.0, margin_pct=0.2),
        repeats=1,
        warmup=0,
    )

    # Avoid the heavy optimisation pipeline inside the agent for unit testing.
    monkeypatch.setattr(
        "agnitra.core.runtime.agent._optimize_model",
        lambda model, tensor, enable_rl=True: model,
    )

    def _fake_profile(module, tensor):
        return {"gpu": {"gpu_utilisation": 33.3}}

    monkeypatch.setattr("agnitra.core.runtime.agent.profile_model", _fake_profile)

    module = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    sample = torch.randn(1, 4)

    result = agent.optimize(
        module,
        sample,
        project_id="unit-tests",
        model_name="toy",
        enable_rl=False,
        metadata={"suite": "runtime-agent"},
    )

    assert result.usage_event is not None
    assert result.usage_event.project_id == "unit-tests"
    assert result.usage_event.model_name == "toy"
    assert agent.usage_meter.all_events()
    assert result.baseline.latency_ms >= 0.0
    assert result.optimized.latency_ms >= 0.0
    assert isinstance(result.optimized.telemetry, dict)
    assert result.optimized.gpu_utilization == pytest.approx(33.3)
