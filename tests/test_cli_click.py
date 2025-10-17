from pathlib import Path
from types import SimpleNamespace

import torch
from click.testing import CliRunner

from agnitra.cli import cli as cli_group


def _create_script_module(path: Path) -> None:
    script = torch.jit.trace(lambda x: x * 2, torch.randn(1))
    script.save(str(path))


def test_optimize_command_creates_output(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.pt"
    _create_script_module(model_path)

    # Avoid running the heavy optimization pipeline during CLI tests

    def _fake_optimize(model, **kwargs):
        return SimpleNamespace(optimized_model=model, usage_event=None, notes={})

    monkeypatch.setattr("agnitra.cli.optimize_model", _fake_optimize)
    monkeypatch.setattr("agnitra.cli.optimize_with_metering", _fake_optimize)

    runner = CliRunner()
    result = runner.invoke(
        cli_group,
        ["optimize", "--model", str(model_path), "--input-shape", "1"],
    )

    assert result.exit_code == 0, result.output
    optimized = tmp_path / "demo_optimized.pt"
    assert optimized.exists(), "Expected optimized model artifact"
    assert "Optimized model written" in result.output


def test_optimize_command_custom_output(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.pt"
    output_path = tmp_path / "custom.pt"
    _create_script_module(model_path)

    monkeypatch.setattr("agnitra.cli.optimize_model", lambda model, **_: SimpleNamespace(optimized_model=model, usage_event=None, notes={}))
    monkeypatch.setattr("agnitra.cli.optimize_with_metering", lambda model, **_: SimpleNamespace(optimized_model=model, usage_event=None, notes={}))

    runner = CliRunner()
    result = runner.invoke(
        cli_group,
        [
            "optimize",
            "--model",
            str(model_path),
            "--input-shape",
            "1",
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()


def test_optimize_command_passes_license_flags(tmp_path, monkeypatch):
    model_path = tmp_path / "demo.pt"
    _create_script_module(model_path)

    captured = {}

    def _fake_optimize(model, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(optimized_model=model, usage_event=None, notes={})

    monkeypatch.setattr("agnitra.cli.optimize_model", _fake_optimize)
    monkeypatch.setattr("agnitra.cli.optimize_with_metering", _fake_optimize)

    runner = CliRunner()
    result = runner.invoke(
        cli_group,
        [
            "optimize",
            "--model",
            str(model_path),
            "--input-shape",
            "1",
            "--target",
            "A100",
            "--offline",
            "--require-license",
            "--license-seat",
            "seat-007",
            "--license-org",
            "enterprise-org",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["offline"] is True
    assert captured["require_license"] is True
    assert captured["license_seat"] == "seat-007"
    assert captured["license_org_id"] == "enterprise-org"
    assert captured["metadata"]["target"] == "A100"
