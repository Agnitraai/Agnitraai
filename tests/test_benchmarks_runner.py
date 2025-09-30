import json

from pathlib import Path

import pytest

from agnitra.benchmarks import runner as bench_runner


class _DummyModel:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, x):  # pragma: no cover - trivial
        self.calls += 1
        return x


def test_run_benchmark_generates_artifacts(tmp_path, monkeypatch):
    dummy_model = _DummyModel()

    monkeypatch.setattr(bench_runner, "optimize_model", lambda *args, **kwargs: dummy_model)
    monkeypatch.setattr(bench_runner, "torch", None)

    summary = bench_runner.run_benchmark(
        dummy_model,
        input_tensor={"agnitra_token_count": 32},
        out_dir=tmp_path,
        repeats=3,
        warmup=1,
        enable_rl=False,
        token_count=32,
    )

    assert summary["tokens_before_per_sec"] > 0.0
    assert summary["tokens_after_per_sec"] > 0.0
    assert summary["results"]["baseline"]["tokens_per_sec"] == summary["tokens_before_per_sec"]

    before = json.loads(Path(tmp_path, "before.json").read_text(encoding="utf-8"))
    after = json.loads(Path(tmp_path, "after.json").read_text(encoding="utf-8"))
    diff = json.loads(Path(tmp_path, "summary_diff.json").read_text(encoding="utf-8"))
    assert before["tokens_per_sec"] == pytest.approx(summary["tokens_before_per_sec"])
    assert after["tokens_per_sec"] == pytest.approx(summary["tokens_after_per_sec"])
    assert "latency_delta_ms" in diff


def test_run_single_benchmark_writes_csv_and_plots(tmp_path, monkeypatch):
    import benchmark_runner

    summary_stub = {
        "speedup": 1.5,
        "latency_before_ms": 10.0,
        "latency_after_ms": 6.0,
        "memory_before_gb": 0.25,
        "memory_after_gb": 0.2,
        "tokens_before_per_sec": 100.0,
        "tokens_after_per_sec": 180.0,
        "tokens_per_sec_delta": 80.0,
        "results": {
            "baseline": {
                "latency_ms": 10.0,
                "memory_bytes": 100,
                "repeats": 3,
                "tokens_per_sec": 100.0,
            },
            "optimized": {
                "latency_ms": 6.0,
                "memory_bytes": 80,
                "repeats": 3,
                "tokens_per_sec": 180.0,
            },
        },
    }

    monkeypatch.setattr(benchmark_runner, "run_benchmark", lambda *args, **kwargs: summary_stub)
    captured = []
    monkeypatch.setattr(benchmark_runner, "_create_plots", lambda summary, path: captured.append(path))

    result = benchmark_runner.run_single_benchmark(
        model="dummy",
        input_tensor=None,
        out_dir=tmp_path,
        repeats=1,
        warmup=0,
        enable_rl=False,
    )

    assert result is summary_stub
    csv_path = tmp_path / "summary.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "variant,latency_ms,memory_gb,tokens_per_sec"
    assert any("optimized" in line for line in lines[1:])

    details_path = tmp_path / "summary_details.json"
    assert details_path.exists()
    payload = json.loads(details_path.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"baseline", "optimized"}

    assert captured == [tmp_path / "benchmark_plots.png"]
