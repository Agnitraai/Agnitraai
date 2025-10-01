import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

# Ensure the project root is on the import path.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agnitra.sdk.optimizer import (
    collect_telemetry,
    optimize_model,
    request_kernel_suggestions,
    run_rl_tuning,
)


class ToyModel(nn.Module):
    def forward(self, x):
        return x * 2


def test_telemetry_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    def boom(*args, **kwargs):
        raise RuntimeError("telemetry boom")

    monkeypatch.setattr("agnitra.sdk.optimizer.collect_telemetry", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x)
    assert result is model
    assert "Telemetry collection failed" in caplog.text


def test_ir_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra.sdk.optimizer.collect_telemetry", lambda m, t: []
    )

    def boom(*args, **kwargs):
        raise RuntimeError("ir boom")

    monkeypatch.setattr("agnitra.sdk.optimizer.extract_ir", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x)
    assert result is model
    assert "IR extraction failed" in caplog.text


def test_llm_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra.sdk.optimizer.collect_telemetry", lambda m, t: []
    )
    monkeypatch.setattr("agnitra.sdk.optimizer.extract_ir", lambda m, t: [])

    def boom(*args, **kwargs):
        raise RuntimeError("llm boom")

    monkeypatch.setattr(
        "agnitra.sdk.optimizer.request_kernel_suggestions", boom
    )
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x, enable_rl=False)
    assert result is model
    assert "LLM call failed" in caplog.text


def test_rl_failure_returns_baseline(monkeypatch, caplog):
    model = ToyModel()
    x = torch.randn(1)

    monkeypatch.setattr(
        "agnitra.sdk.optimizer.collect_telemetry", lambda m, t: []
    )
    monkeypatch.setattr("agnitra.sdk.optimizer.extract_ir", lambda m, t: [])
    monkeypatch.setattr(
        "agnitra.sdk.optimizer.request_kernel_suggestions",
        lambda t, i, client=None: None,
    )

    def boom(*args, **kwargs):
        raise RuntimeError("rl boom")

    monkeypatch.setattr("agnitra.sdk.optimizer.run_rl_tuning", boom)
    with caplog.at_level(logging.ERROR):
        result = optimize_model(model, x, enable_rl=True)
    assert result is model
    assert "RL tuning failed" in caplog.text


def test_request_kernel_suggestions_requires_openai(monkeypatch):
    def boom():
        raise RuntimeError("openai missing")

    monkeypatch.setattr("agnitra.sdk.optimizer.require_openai", boom)
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

    monkeypatch.setattr("agnitra.sdk.optimizer.require_sb3", boom)
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
        "agnitra.sdk.optimizer.require_sb3", lambda: (DummyPPO, DummyGym())
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
        "agnitra.sdk.optimizer.require_sb3", lambda: (DummyPPO, DummyGym())
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
