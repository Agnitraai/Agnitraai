import torch

from agnitra.sdk import optimize_model as sdk_optimize


class DummyNet(torch.nn.Module):
    def forward(self, x):  # type: ignore[override]
        return x + 1


def test_optimize_model_invokes_backend(monkeypatch):
    model = DummyNet()
    captured = {}

    def _fake_optimize(mod, tensor, enable_rl):
        captured["tensor_shape"] = tuple(tensor.shape)
        captured["enable_rl"] = enable_rl
        return mod

    monkeypatch.setattr("agnitra.sdk._optimize_model", _fake_optimize)

    optimized = sdk_optimize(model, input_shape=(1, 4), enable_rl=False)

    assert optimized is model
    assert captured["tensor_shape"] == (1, 4)
    assert captured["enable_rl"] is False


def test_optimize_model_requires_shape(monkeypatch):
    model = DummyNet()
    monkeypatch.setattr("agnitra.sdk._optimize_model", lambda *a, **k: a[0])

    try:
        sdk_optimize(model)
    except ValueError as exc:
        assert "input_shape" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected ValueError when no input provided")


def test_optimize_model_uses_tensor(monkeypatch):
    model = DummyNet()
    tensor = torch.randn(2, 2)

    def _fake_optimize(mod, sample, enable_rl):
        assert sample is tensor
        return mod

    monkeypatch.setattr("agnitra.sdk._optimize_model", _fake_optimize)

    result = sdk_optimize(model, input_tensor=tensor, input_shape=(4,))
    assert result is model

