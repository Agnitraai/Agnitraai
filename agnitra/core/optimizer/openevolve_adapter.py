"""OpenEvolve integration utilities for Agnitra logs.

This module provides light-weight helpers that take the structured log output
from :class:`~agnitra.core.optimizer.LLMOptimizer` and translate it into an
"evolution" problem consumable by the OpenEvolve library. The integration is
optional – when ``openevolve`` is unavailable the helpers degrade gracefully and
emit informative log messages so callers can fall back to heuristic or LLM-only
strategies.

The primary entry point is :func:`run_open_evolve_from_log`, which loads a log
file, constructs an :class:`OpenEvolveProblem`, and executes a short evolution
loop via :class:`OpenEvolveRunner`. The runner prefers the ``gpt-5-mini`` model
by default to align with Agnitra's LLM optimisation pipeline, but callers can
override the underlying configuration when needed.
"""

from __future__ import annotations

import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

LOGGER = logging.getLogger(__name__)

EvaluatorFn = Callable[[Path], Mapping[str, Any]]
RunEvolutionFn = Callable[..., Any]

_LATENCY_REGEX = re.compile(r"expected_latency_ms\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_TARGET_REGEX = re.compile(r"target_latency_ms\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        a = _to_int(value[0], default[0])
        b = _to_int(value[1], default[1])
        return a, b
    return default


def _render_initial_program(suggestion: Mapping[str, Any], baseline_latency: Optional[float]) -> str:
    block_size = _to_int(suggestion.get("block_size"), 128)
    tile_m, tile_n = _ensure_pair(suggestion.get("tile_shape"), (64, 64))
    unroll = _to_int(suggestion.get("unroll_factor"), 2)
    expected_latency = _to_float(suggestion.get("expected_latency_ms"))
    if expected_latency is None:
        expected_latency = baseline_latency or 7.5
    rationale = suggestion.get("rationale") or "Seed configuration derived from Agnitra log."
    program = f"""
# Auto-generated seed program from Agnitra log analysis
BLOCK_SIZE = {block_size}
TILE_M = {tile_m}
TILE_N = {tile_n}
UNROLL_FACTOR = {unroll}
EXPECTED_LATENCY_MS = {expected_latency:.4f}


def candidate_configuration():
    return {{
        "block_size": BLOCK_SIZE,
        "tile_shape": (TILE_M, TILE_N),
        "unroll_factor": UNROLL_FACTOR,
        "expected_latency_ms": EXPECTED_LATENCY_MS,
        "rationale": {json.dumps(str(rationale))}
    }}
"""
    return textwrap.dedent(program).strip()


def _extract_expected_latency(text: str) -> Optional[float]:
    try:
        payload = json.loads(text)
    except Exception:
        payload = None
    if isinstance(payload, Mapping):
        for key in ("expected_latency_ms", "target_latency_ms"):
            value = payload.get(key)
            extracted = _to_float(value)
            if extracted is not None:
                return extracted
    match = _LATENCY_REGEX.search(text)
    if match:
        return float(match.group(1))
    match = _TARGET_REGEX.search(text)
    if match:
        return float(match.group(1))
    return None


@dataclass(frozen=True)
class OpenEvolveConfig:
    """Configuration controlling the OpenEvolve invocation."""

    iterations: int = 120
    llm_model: str = "gpt-5-mini"
    extra_config: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OpenEvolveProblem:
    """Normalised optimisation problem derived from an Agnitra log."""

    name: str
    description: str
    baseline_latency_ms: Optional[float]
    bottleneck_op: Optional[str]
    target_latency_ms: Optional[float]
    suggestion: Mapping[str, Any]
    initial_program: str
    iterations: Optional[int] = None
    telemetry_summary: Mapping[str, Any] = field(default_factory=dict)
    raw_report: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_report(
        cls,
        payload: Mapping[str, Any],
        *,
        name: Optional[str] = None,
    ) -> "OpenEvolveProblem":
        report = payload.get("report", {})
        baseline = report.get("baseline", {})
        best = report.get("best_model", {})
        suggestion = best.get("suggestion", {}) or {}
        baseline_latency = _to_float(baseline.get("latency_ms"))
        expected = _to_float(suggestion.get("expected_latency_ms"))
        target = _to_float(suggestion.get("target_latency_ms"))
        if target is None:
            target = expected
        if target is None and baseline_latency is not None:
            target = baseline_latency * 0.85
        initial_program = _render_initial_program(suggestion, baseline_latency)
        problem_name = name or str(payload.get("name") or best.get("model") or "agnitra-log")
        description = payload.get("description") or (
            f"Baseline latency {baseline_latency} ms for op {baseline.get('op')}"
        )
        iterations = payload.get("iterations")
        if isinstance(iterations, str) and iterations.isdigit():
            iterations = int(iterations)
        elif isinstance(iterations, (int, float)):
            iterations = int(iterations)
        else:
            iterations = None
        return cls(
            name=problem_name,
            description=str(description),
            baseline_latency_ms=baseline_latency,
            bottleneck_op=baseline.get("op"),
            target_latency_ms=target,
            suggestion=dict(suggestion),
            initial_program=initial_program,
            iterations=iterations,
            telemetry_summary=dict(baseline),
            raw_report=dict(payload),
        )

    def default_iterations(self, fallback: int = 120) -> int:
        if self.iterations:
            return max(1, int(self.iterations))
        return fallback

    def build_default_evaluator(self) -> EvaluatorFn:
        baseline = self.baseline_latency_ms or 10.0
        target = self.target_latency_ms or max(baseline - 1.0, baseline * 0.85)

        def evaluator(path: Path) -> Dict[str, Any]:
            try:
                text = Path(path).read_text()
            except Exception:
                text = ""
            candidate_latency = _extract_expected_latency(text) or target
            improvement = baseline - candidate_latency
            return {
                "score": improvement,
                "baseline_latency_ms": baseline,
                "candidate_latency_ms": candidate_latency,
            }

        return evaluator


@dataclass(frozen=True)
class OpenEvolveResult:
    """Result summary for an OpenEvolve optimisation run."""

    best_code: Optional[str]
    metrics: Mapping[str, Any]
    raw: Any
    iterations: int

    @classmethod
    def from_raw(
        cls,
        raw: Any,
        *,
        iterations: int,
    ) -> "OpenEvolveResult":
        best_code: Optional[str] = None
        metrics: Dict[str, Any] = {"iterations": iterations}
        if raw is not None:
            if isinstance(raw, Mapping):
                candidate = raw.get("best_code")
                if isinstance(candidate, str):
                    best_code = candidate
                for key in ("best_score", "metrics", "history"):
                    if key in raw:
                        metrics[key] = raw[key]
            else:
                best_code = getattr(raw, "best_code", None)
                if hasattr(raw, "best_score"):
                    metrics["best_score"] = getattr(raw, "best_score")
                if hasattr(raw, "metrics"):
                    metrics["metrics"] = getattr(raw, "metrics")
        return cls(best_code=best_code, metrics=metrics, raw=raw, iterations=iterations)


class OpenEvolveRunner:
    """Helper that executes OpenEvolve when available."""

    def __init__(
        self,
        *,
        config: Optional[OpenEvolveConfig] = None,
        run_evolution: Optional[RunEvolutionFn] = None,
    ) -> None:
        self._config = config or OpenEvolveConfig()
        self._run_evolution = run_evolution or self._import_runner()

    def _import_runner(self) -> Optional[RunEvolutionFn]:
        try:
            from openevolve import run_evolution  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            LOGGER.debug("OpenEvolve unavailable: %s", exc)
            return None
        return run_evolution

    def available(self) -> bool:
        return self._run_evolution is not None

    def optimize(
        self,
        problem: OpenEvolveProblem,
        *,
        iterations: Optional[int] = None,
        evaluator: Optional[EvaluatorFn] = None,
        extra_config: Optional[Mapping[str, Any]] = None,
    ) -> OpenEvolveResult:
        if self._run_evolution is None:
            raise RuntimeError("OpenEvolve is not installed; cannot run evolution")

        run_iterations = iterations or problem.default_iterations(self._config.iterations)
        eval_fn = evaluator or problem.build_default_evaluator()
        config_payload: Dict[str, Any] = dict(self._config.extra_config)
        if self._config.llm_model:
            llm_settings = config_payload.get("llm")
            if isinstance(llm_settings, Mapping):
                if "model" not in llm_settings:
                    llm_settings = dict(llm_settings)
                    llm_settings["model"] = self._config.llm_model
            else:
                llm_settings = {"model": self._config.llm_model}
            config_payload["llm"] = llm_settings
        if extra_config:
            config_payload.update(extra_config)

        run_kwargs: Dict[str, Any] = {
            "initial_program": problem.initial_program,
            "evaluator": eval_fn,
            "iterations": run_iterations,
        }
        if config_payload:
            run_kwargs["config"] = config_payload

        LOGGER.debug(
            "Running OpenEvolve for problem '%s' with iterations=%s and model=%s",
            problem.name,
            run_iterations,
            self._config.llm_model,
        )
        raw_result = self._run_evolution(**run_kwargs)
        return OpenEvolveResult.from_raw(raw_result, iterations=run_iterations)


def load_openevolve_problem(path: str | Path) -> OpenEvolveProblem:
    """Load an :class:`OpenEvolveProblem` from a JSON log file."""

    log_path = Path(path)
    payload = json.loads(log_path.read_text())
    return OpenEvolveProblem.from_report(payload, name=log_path.stem)


def run_open_evolve_from_log(
    path: str | Path,
    *,
    runner: Optional[OpenEvolveRunner] = None,
    config: Optional[OpenEvolveConfig] = None,
    iterations: Optional[int] = None,
    evaluator: Optional[EvaluatorFn] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> Optional[OpenEvolveResult]:
    """Convenience wrapper that executes OpenEvolve for a given log file.

    When OpenEvolve is not installed, ``None`` is returned and a debug log entry
    is emitted; this allows callers to treat the integration as best-effort.
    """

    runner = runner or OpenEvolveRunner(config=config)
    if not runner.available():
        LOGGER.info("OpenEvolve not available; skipping optimisation for %s", path)
        return None
    problem = load_openevolve_problem(path)
    return runner.optimize(
        problem,
        iterations=iterations,
        evaluator=evaluator,
        extra_config=extra_config,
    )
