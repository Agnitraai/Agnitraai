import json
import torch

import agnitra.telemetry_collector as tc


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_profile_model(tmp_path):
    model = ToyModel().eval()
    x = torch.randn(1, 4)
    out_file = tmp_path / "telemetry.json"
    data = tc.profile_model(model, x, str(out_file))
    assert "events" in data and "gpu" in data
    first = data["events"][0]
    assert "cuda_time_total" in first
    assert "input_shapes" in first
    assert "self_cpu_memory_usage" in first
    loaded = json.loads(out_file.read_text())
    assert loaded["events"]


def test_capture_gpu_metrics(monkeypatch):
    class DummyUtil:
        gpu = 77
        memory = 55

    def dummy_handle(idx):
        return idx

    monkeypatch.setattr(tc, "_NVML_AVAILABLE", True)
    monkeypatch.setattr(tc, "nvmlInit", lambda: None, raising=False)
    monkeypatch.setattr(tc, "nvmlShutdown", lambda: None, raising=False)
    monkeypatch.setattr(tc, "nvmlDeviceGetHandleByIndex", dummy_handle, raising=False)
    monkeypatch.setattr(
        tc,
        "nvmlDeviceGetUtilizationRates",
        lambda h: DummyUtil(),
        raising=False,
    )
    monkeypatch.setattr(
        tc, "nvmlDeviceGetPowerUsage", lambda h: 123000, raising=False
    )

    gpu = tc._capture_gpu_metrics()
    assert gpu.gpu_utilisation == 77
    assert gpu.memory_utilisation == 55
    assert gpu.power_watts == 123.0


def test_profile_model_without_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "torch", None)
    monkeypatch.setattr(tc, "profile", None)
    monkeypatch.setattr(tc, "ProfilerActivity", None)

    out_file = tmp_path / "telemetry.json"
    payload = tc.profile_model(None, None, str(out_file))
    assert payload == {"events": [], "gpu": {}}
    loaded = json.loads(out_file.read_text())
    assert loaded == payload
