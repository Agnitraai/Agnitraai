import sys
from pathlib import Path

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agnitra.demo import demo


def test_main_runs_pipeline(monkeypatch, capsys):
    """demo.main should run DemoNet pipeline when no flags are given."""
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    monkeypatch.setattr("agnitra.demo.demo.DemoNet.optimize", lambda self: "Patched<demo>")
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    demo.main()

    captured = capsys.readouterr()
    assert "Patched<demo>" in captured.out
    assert "CUDA not available" in captured.out


def test_main_profile_samples(monkeypatch):
    """demo.main should invoke profile_sample_models when flag is set."""
    calls = []

    def fake_profile():
        calls.append("called")

    monkeypatch.setattr(sys, "argv", ["demo.py", "--profile-samples"])
    monkeypatch.setattr("agnitra.demo.demo.profile_sample_models", fake_profile)

    demo.main()

    assert calls == ["called"]
