import json
from typing import Dict, List

import pytest

torch = pytest.importorskip("torch")

from agnitra.core.metering import UsageMeter
from agnitra.core.runtime.agent import (
    OptimizationSnapshot,
    RuntimeOptimizationAgent,
)
from agnitra.core.runtime.control_plane import OptimizationPolicy


class _StubTelemetryClient:
    def __init__(self) -> None:
        self.events: List[Dict[str, object]] = []

    def emit(self, event: Dict[str, object]) -> None:
        # store a deep copy to avoid accidental mutation after emission
        self.events.append(json.loads(json.dumps(event)))

    def close(self) -> None:  # pragma: no cover - interface parity
        pass


def _make_snapshot(latency_ms: float, tokens_per_sec: float, gpu_util: float) -> OptimizationSnapshot:
    tokens_processed = 1000
    return OptimizationSnapshot(
        latency_ms=latency_ms,
        tokens_per_sec=tokens_per_sec,
        tokens_processed=tokens_processed,
        gpu_utilization=gpu_util,
        telemetry={"stage": "snapshot"},
        metadata={"stage": "snapshot"},
    )


def test_runtime_usage_event_snapshot(monkeypatch):
    policy = OptimizationPolicy(
        policy_id="snapshot-policy",
        plan_objective="throughput",
        enable_llm=False,
        enable_rl=False,
        calibration_iterations=8,
        calibration_warmup=2,
        telemetry_sample_rate=0.5,
        default_preset={"allow_tf32": True},
        pass_presets=[{"torch_compile": True}],
    )

    baseline_snapshot = _make_snapshot(latency_ms=10.0, tokens_per_sec=1000.0, gpu_util=0.5)
    optimized_snapshot = _make_snapshot(latency_ms=7.0, tokens_per_sec=1500.0, gpu_util=0.7)

    snapshots = [baseline_snapshot, optimized_snapshot]

    def _fake_capture(self, module, tensor, torch_mod, *, stage, repeats, warmup, extra_metadata):
        return snapshots.pop(0)

    monkeypatch.setattr(RuntimeOptimizationAgent, "_capture_snapshot", _fake_capture)

    def _fake_optimize_model(model, tensor, enable_rl=True, *, preset=None, context=None, policy=None):
        if context is not None:
            if preset:
                context["applied_preset"] = dict(preset)
                context["applied_preset_source"] = "preset_override"
            pass_presets = (policy or {}).get("pass_presets") or []
            if pass_presets:
                context["applied_pass_presets"] = list(pass_presets)
            context.setdefault("telemetry_event_count", 0)
            context.setdefault("ir_node_count", 0)
        return model

    monkeypatch.setattr("agnitra.core.runtime.agent._optimize_model", _fake_optimize_model)

    telemetry_client = _StubTelemetryClient()
    meter = UsageMeter(rate_per_gpu_hour=5.0, margin_pct=0.2)
    agent = RuntimeOptimizationAgent(usage_meter=meter, repeats=1, warmup=0)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    sample = torch.randn(1, 4)

    fingerprint = {
        "model_name": "DemoModel",
        "param_count": 42,
        "buffer_count": 0,
        "framework": {"framework": "torch", "version": "2.x", "backend": "cuda"},
        "gpu": {"vendor": "nvidia", "model": "A100"},
        "input_signature": {"shape": [1, 4], "dtype": "torch.float32", "device": "cuda:0"},
        "host": "unit-test",
    }

    result = agent.optimize(
        model,
        sample,
        project_id="proj-snapshot",
        model_name="DemoModel",
        enable_rl=False,
        metadata={"custom": "value"},
        policy=policy,
        cached_profile=None,
        telemetry_client=telemetry_client,
        fingerprint=fingerprint,
        fingerprint_signature="fingerprinted",
        cache_signature="fingerprinted",
    )

    assert result.usage_event is not None
    assert len(telemetry_client.events) == 1

    event = telemetry_client.events[0]
    for dynamic_key in ("event_id", "ts", "emitted_at", "idempotency_key", "run_id"):
        event.pop(dynamic_key, None)
    event["context"].pop("invoked_at", None)

    expected_gpu_hours = (1000 / 1500) / 3600.0
    expected_gpu_hours_before = (1000 / 1000) / 3600.0
    expected_cost_after = expected_gpu_hours * 5.0
    expected_cost_before = expected_gpu_hours_before * 5.0

    assert event["project_id"] == "proj-snapshot"
    assert event["event"] == "usage"
    assert event["model_name"] == "DemoModel"
    assert event["plan"] == {"objective": "throughput", "sample_rate": 0.5}
    assert event["sig"] == "fingerprinted"
    assert event["gpu"] == {"vendor": "nvidia", "model": "A100"}
    assert event["workload"] == {
        "device": "cuda:0",
        "dtype": "torch.float32",
        "framework": "torch",
        "input_shape": [1, 4],
        "model": "DemoModel",
        "precision": None,
    }

    baseline_metrics = event["metrics"]["baseline"]
    optimized_metrics = event["metrics"]["optimized"]
    assert baseline_metrics == {
        "tokens_per_s": 1000.0,
        "latency_ms": 10.0,
        "gpu_util": 0.5,
    }
    assert optimized_metrics == {
        "tokens_per_s": 1500.0,
        "latency_ms": 7.0,
        "gpu_util": 0.7,
    }

    computed = event["computed"]
    assert computed["tokens"] == 1000
    assert computed["uplift_pct"] == pytest.approx(50.0)
    assert computed["gpu_hours"] == pytest.approx(expected_gpu_hours)

    billing = event["billing"]
    assert billing["gpu_hours_after"] == pytest.approx(expected_gpu_hours)
    assert billing["gpu_hours_before"] == pytest.approx(expected_gpu_hours_before)
    assert billing["gpu_hours_saved"] == pytest.approx(expected_gpu_hours_before - expected_gpu_hours)
    assert billing["usage_charge"] == pytest.approx(expected_cost_after)
    expected_success_fee = (expected_cost_before - expected_cost_after) * 0.2
    assert billing["success_fee"] == pytest.approx(expected_success_fee)
    assert billing["total_billable"] == pytest.approx(expected_cost_after + expected_success_fee)

    context = event["context"]
    assert context["cache_hit"] is False
    assert context["policy_id"] == "snapshot-policy"
    assert context["fingerprint_signature"] == "fingerprinted"
    assert context["preset_source"] == "policy"
    assert context["rl_enabled"] is False
    assert context["auto_retrain_requested"] is False
    assert context["auto_retrain_scheduled"] is False
    assert context["abtest_repeats"] == 8
    assert context["abtest_warmup"] == 2
    assert context["applied_preset"] == {"allow_tf32": True}
    assert context["applied_preset_source"] == "preset_override"
    assert context["applied_pass_presets"] == [{"torch_compile": True}]
    assert context["policy_pass_presets"] == [{"torch_compile": True}]
    assert context["telemetry_event_count"] == 0
    assert context["ir_node_count"] == 0


def test_auto_retrain_job_payload(monkeypatch):
    jobs: List[Dict[str, object]] = []

    def _record_auto_retrain(self, model, tensor, torch_mod, policy_payload):
        jobs.append(dict(policy_payload))

    monkeypatch.setattr(RuntimeOptimizationAgent, "_schedule_auto_retrain", _record_auto_retrain)

    policy = OptimizationPolicy(
        policy_id="auto-policy",
        plan_objective="efficiency",
        enable_llm=False,
        enable_rl=False,
        calibration_iterations=2,
        calibration_warmup=1,
        telemetry_sample_rate=1.0,
        default_preset={"allow_tf32": True},
        auto_retrain=True,
        auto_retrain_interval=0.3,
    )

    baseline_snapshot = _make_snapshot(latency_ms=5.0, tokens_per_sec=500.0, gpu_util=0.4)
    optimized_snapshot = _make_snapshot(latency_ms=4.0, tokens_per_sec=600.0, gpu_util=0.5)
    snapshots = [baseline_snapshot, optimized_snapshot]

    def _fake_capture(self, module, tensor, torch_mod, *, stage, repeats, warmup, extra_metadata):
        return snapshots.pop(0)

    monkeypatch.setattr(RuntimeOptimizationAgent, "_capture_snapshot", _fake_capture)
    monkeypatch.setattr("agnitra.core.runtime.agent._optimize_model", lambda model, tensor, **_: model)

    telemetry_client = _StubTelemetryClient()
    agent = RuntimeOptimizationAgent(usage_meter=UsageMeter(), repeats=1, warmup=0)

    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    sample = torch.randn(1, 4)
    fingerprint = {"model_name": "AutoModel", "gpu": {}, "framework": {}, "input_signature": {}}

    agent.optimize(
        model,
        sample,
        project_id="auto-proj",
        model_name="AutoModel",
        enable_rl=False,
        policy=policy,
        telemetry_client=telemetry_client,
        fingerprint=fingerprint,
        fingerprint_signature="auto-sig",
        cache_signature="auto-sig",
    )

    assert len(jobs) == 1
    job = jobs[0]
    assert job["policy_id"] == "auto-policy"
    assert job["project_id"] == "auto-proj"
    assert job["model_name"] == "AutoModel"
    assert job["fingerprint_signature"] == "auto-sig"
    assert job["auto_retrain_interval"] == 0.3
