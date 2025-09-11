import sys
from pathlib import Path

# Ensure project root on sys.path for direct module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from agnitra.demo import DemoNet


def test_demo_net_optimize(monkeypatch):
    """Integration test for DemoNet.optimize."""
    # Mock potential heavy or network-bound operations
    monkeypatch.setattr(
        "agnitra.sdk.optimizer.request_kernel_suggestions",
        lambda *args, **kwargs: "mocked",
    )
    monkeypatch.setattr(
        "agnitra.sdk.optimizer.run_rl_tuning",
        lambda *args, **kwargs: None,
    )

    net = DemoNet()
    result = net.optimize(model="dummy-model")
    assert result.startswith("Patched<")
