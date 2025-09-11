import torch
from cli.main import main


def _make_dummy_model(path):
    script = torch.jit.trace(lambda x: x * 2, torch.randn(1))
    script.save(str(path))


def test_profile_missing_model(tmp_path, capsys):
    missing = tmp_path / "missing.pt"
    exit_code = main(["profile", str(missing)])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not found" in captured.out


def test_profile_success(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.pt"
    _make_dummy_model(model_path)

    output_path = tmp_path / "telemetry.json"
    # Avoid heavy profiling by mocking the collector
    monkeypatch.setattr("cli.main.profile_model", lambda *a, **k: None)

    exit_code = main([
        "profile",
        str(model_path),
        "--input-shape",
        "1",
        "--output",
        str(output_path),
    ])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Telemetry written" in captured.out
