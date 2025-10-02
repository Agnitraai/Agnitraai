import torch
from torch import nn

from agnitra._sdk import optimizer


def test_collect_telemetry_handles_missing_cuda(monkeypatch):
    class DummyEvent:
        key = "forward"
        cpu_time_total = 1000.0
        input_shapes = []
        self_cpu_memory_usage = 0
        # Note: no cuda_time_total or self_cuda_memory_usage

    class DummyProfile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def key_averages(self):
            return [DummyEvent()]

    class DummyRecord:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(optimizer, "profile", lambda *a, **k: DummyProfile())
    monkeypatch.setattr(optimizer, "record_function", DummyRecord)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    model = nn.Linear(1, 1)
    x = torch.randn(1, 1)

    telemetry = optimizer.collect_telemetry(model, x)
    assert telemetry == [
        {
            "name": "forward",
            "cpu_time_ms": 1000.0 / 1e6,
            "cuda_time_ms": 0.0,
            "input_shape": [],
            "cpu_memory_bytes": 0,
            "cuda_memory_bytes": 0,
        }
    ]
