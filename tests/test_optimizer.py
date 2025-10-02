import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import pytest
import torch
from torch import nn

# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agnitra._sdk.optimizer import (
    collect_telemetry,
    optimize_log_with_open_evolve,
    optimize_model,
    request_kernel_suggestions,
    run_rl_tuning,
)
from agnitra.core.optimizer import OpenEvolveRunner


class ToyModel(nn.Module):
    def forward(self, x):
        return x * 2


def test_telemetry_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    def boom(*args, **kwargs):
        raise RuntimeError("telemetry boom")

    monkeypatch.setattr("agnitra._sdk.optimizer.collect_telemetry", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x)
    assert result is model
    assert "Telemetry collection failed" in caplog.text


def test_ir_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.collect_telemetry", lambda m, t: []
    )

    def boom(*args, **kwargs):
        raise RuntimeError("ir boom")

    monkeypatch.setattr("agnitra._sdk.optimizer.extract_ir", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x)
    assert result is model
    assert "IR extraction failed" in caplog.text


def test_llm_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.collect_telemetry", lambda m, t: []
    )
    monkeypatch.setattr("agnitra._sdk.optimizer.extract_ir", lambda m, t: [])

    def boom(*args, **kwargs):
        raise RuntimeError("llm boom")

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.request_kernel_suggestions", boom
    )
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x, enable_rl=False)
    assert result is model
    assert "LLM call failed" in caplog.text


def test_rl_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.collect_telemetry", lambda m, t: []
    )
    monkeypatch.setattr("agnitra._sdk.optimizer.extract_ir", lambda m, t: [])
    monkeypatch.setattr(
        "agnitra._sdk.optimizer.request_kernel_suggestions",
        lambda t, i, client=None: None,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("rl boom")

    monkeypatch.setattr("agnitra._sdk.optimizer.run_rl_tuning", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x, enable_rl=True)
    assert result is model
    assert "RL tuning failed" in caplog.text


def test_request_kernel_suggestions_requires_openai(monkeypatch):
    def boom():
        raise RuntimeError("openai missing")

    monkeypatch.setattr("agnitra._sdk.optimizer.require_openai", boom)
    assert request_kernel_suggestions([], []) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_collect_telemetry_aligns_to_model_device(tmp_path):
    class Dummy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert x.device == self.proj.weight.device
            return self.proj(x)

    model = Dummy().cuda()
    input_cpu = torch.randn(2, 8)
    telemetry = collect_telemetry(model, input_cpu)
    assert isinstance(telemetry, list)
    assert telemetry  # profiler should record at least one event


def test_run_rl_tuning_requires_sb3(monkeypatch):
    def boom():
        raise RuntimeError("sb3 missing")

    monkeypatch.setattr("agnitra._sdk.optimizer.require_sb3", boom)
    run_rl_tuning([], [])  # Should not raise


def test_run_rl_tuning_missing_env_logs_warning(monkeypatch, caplog):
    class DummyPPO:
        def __init__(self, *args, **kwargs):
            pass

        def learn(self, total_timesteps):  # pragma: no cover - simplicity
            pass

    class DummyGym:
        def make(self, *args, **kwargs):
            raise Exception("env missing")

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.require_sb3", lambda: (DummyPPO, DummyGym())
    )
    with caplog.at_level(logging.WARNING):
        run_rl_tuning([], [])
    assert "Gym environment" in caplog.text


def test_run_rl_tuning_uses_cuda_if_available_and_closes_env(monkeypatch):
    captured: Dict[str, Any] = {}

    class DummyPPO:
        def __init__(self, *args, **kwargs):
            captured["device"] = kwargs.get("device")

        def learn(self, total_timesteps):  # pragma: no cover - simplicity
            pass

    class DummyEnv:
        def __init__(self) -> None:
            self.closed = False

        def close(self):
            self.closed = True

    dummy_env = DummyEnv()

    class DummyGym:
        def make(self, *args, **kwargs):
            return dummy_env

    monkeypatch.setattr(
        "agnitra._sdk.optimizer.require_sb3", lambda: (DummyPPO, DummyGym())
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    run_rl_tuning([], [])
    assert captured["device"] == "cuda"
    assert dummy_env.closed


def test_request_kernel_suggestions_handles_empty_response():
    class DummyClient:
        class Responses:
            def create(self, *args, **kwargs):  # pragma: no cover - simplicity
                class Response:
                    pass

                return Response()

        responses = Responses()

    client = DummyClient()
    assert request_kernel_suggestions([], [], client=client) is None


def test_request_kernel_suggestions_handles_minimal_output():
    class DummyClient:
        class Entry:
            pass

        class Item:
            pass

        class Response:
            pass

        class Responses:
            def create(self, *args, **kwargs):  # pragma: no cover - simplicity
                return DummyClient.Response()

        responses = Responses()

    DummyClient.Item.content = [DummyClient.Entry()]
    DummyClient.Response.output = [DummyClient.Item()]
    client = DummyClient()
    assert request_kernel_suggestions([], [], client=client) is None

def _sample_log_payload() -> Dict[str, Any]:
    return {
        "report": {
            "baseline": {"latency_ms": 10.0, "op": "matmul"},
            "best_model": {
                "model": "gpt-5-mini",
                "suggestion": {
                    "block_size": 128,
                    "tile_shape": [64, 64],
                    "unroll_factor": 2,
                    "expected_latency_ms": 7.4,
                },
            },
        }
    }


def test_optimize_log_with_open_evolve_invokes_runner(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(_sample_log_payload()))
    capture: Dict[str, Any] = {}

    def fake_run(**kwargs):
        capture.update(kwargs)
        candidate_file = tmp_path / "candidate.json"
        candidate_file.write_text(json.dumps({"expected_latency_ms": 6.8}))
        kwargs["evaluator"](candidate_file)
        return SimpleNamespace(best_code="print('ok')", best_score=1.0)

    runner = OpenEvolveRunner(run_evolution=fake_run)
    result = optimize_log_with_open_evolve(log_path, runner=runner, iterations=8)
    assert result is not None
    assert result.best_code == "print('ok')"
    assert capture["iterations"] == 8
    assert capture["config"]["llm"]["model"] == "gpt-5-mini"


def test_optimize_log_with_open_evolve_missing_file_returns_none(tmp_path, caplog):
    missing = tmp_path / "absent.json"
    with caplog.at_level(logging.ERROR):
        result = optimize_log_with_open_evolve(missing)
    assert result is None
    assert "OpenEvolve log file not found" in caplog.text
