from pathlib import Path

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
    monkeypatch.setattr("agnitra.cli.optimize_model", lambda model, **_: model)

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

    monkeypatch.setattr("agnitra.cli.optimize_model", lambda model, **_: model)

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
