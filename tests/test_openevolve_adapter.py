import json
from types import SimpleNamespace

import pytest

from agnitra.core.optimizer import (
    OpenEvolveProblem,
    OpenEvolveRunner,
    run_open_evolve_from_log,
)


def _sample_report() -> dict:
    return {
        "report": {
            "baseline": {"latency_ms": 12.5, "op": "matmul"},
            "best_model": {
                "model": "gpt-5-mini",
                "suggestion": {
                    "block_size": 256,
                    "tile_shape": [64, 128],
                    "unroll_factor": 4,
                    "expected_latency_ms": 8.4,
                    "rationale": "seed",
                },
            },
        }
    }


def test_open_evolve_problem_from_report_generates_program(tmp_path):
    problem = OpenEvolveProblem.from_report(_sample_report(), name="sample")
    assert problem.name == "sample"
    assert "# EVOLVE-BLOCK-START" in problem.initial_program
    assert "candidate_configuration" in problem.initial_program
    evaluator = problem.build_default_evaluator()
    candidate_file = tmp_path / "candidate.json"
    candidate_file.write_text(json.dumps({"expected_latency_ms": 7.9}))
    metrics = evaluator(candidate_file)
    if hasattr(metrics, "metrics"):
        metrics = metrics.metrics
    assert pytest.approx(metrics["combined_score"], rel=1e-3) == 12.5 - 7.9


def test_open_evolve_runner_requires_dependency():
    problem = OpenEvolveProblem.from_report(_sample_report())
    runner = OpenEvolveRunner(run_evolution=None)
    assert runner.available() is False
    with pytest.raises(RuntimeError):
        runner.optimize(problem)


def test_open_evolve_runner_invokes_underlying(tmp_path):
    problem = OpenEvolveProblem.from_report(_sample_report())
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        candidate = tmp_path / "candidate.json"
        candidate.write_text(json.dumps({"expected_latency_ms": 7.2}))
        kwargs["evaluator"](candidate)
        return SimpleNamespace(best_code="optimized", best_score=3.3)

    runner = OpenEvolveRunner(run_evolution=fake_run)
    result = runner.optimize(problem, iterations=42)
    assert result.best_code == "optimized"
    assert result.metrics["best_score"] == 3.3
    assert captured["iterations"] == 42
    assert captured["config"]["llm"]["model"] == "gpt-5-mini"


def test_run_open_evolve_from_log_handles_unavailable_runner(tmp_path):
    log_path = tmp_path / "log.json"
    log_path.write_text(json.dumps(_sample_report()))
    runner = OpenEvolveRunner(run_evolution=None)
    assert run_open_evolve_from_log(log_path, runner=runner) is None
