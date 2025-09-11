"""Command line interface for Agnitra.

The CLI is designed for resilience: heavy dependencies like ``torch`` are
imported lazily and all commands return an exit code instead of raising
exceptions.  This allows callers to handle errors gracefully and enables
"self-healing" behaviours in higher level orchestrators.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pathlib import Path as _Path

# Quiet noisy third-party libraries (TF, Gym) for clean CLI output
import os as _os
import warnings as _warnings
import logging as _logging
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow C++ logs
_os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")  # Abseil (TF) logs
_os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")  # legacy Gym warnings
_warnings.filterwarnings("ignore")
_logging.getLogger("absl").setLevel(_logging.ERROR)

# Testing hook: allows tests to monkeypatch a lightweight profiler
profile_model = None  # type: ignore[assignment]


def _parse_shape(s: str) -> Sequence[int]:
    """Parse a comma separated shape string into a sequence of ints."""

    return tuple(int(x) for x in s.split(","))


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Splitting parser construction allows tests to exercise argument handling
    without executing command logic.
    """

    parser = argparse.ArgumentParser(prog="agnitra")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prof = sub.add_parser("profile", help="Profile a Torch model")
    prof.add_argument("model", type=Path, help="Path to a TorchScript model")
    prof.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Comma separated input tensor shape",
    )
    prof.add_argument(
        "--output",
        default="telemetry.json",
        help="Path to write telemetry JSON",
    )

    bench = sub.add_parser("benchmark", help="Benchmark baseline vs optimized model")
    bench.add_argument("model", type=Path, help="Path to a TorchScript model")
    bench.add_argument("--input-shape", default="1,3,224,224", help="Comma separated input tensor shape")
    bench.add_argument("--repeats", type=int, default=10, help="Number of timed iterations")
    bench.add_argument("--warmup", type=int, default=2, help="Number of warmup runs")
    bench.add_argument("--output-dir", default="benchmarks", help="Directory to write JSON outputs")
    bench.add_argument("--llm-rl", action="store_true", help="Enable Codex/GPT-guided preset application")
    bench.add_argument("--only-llm", action="store_true", help="Use LLM preset only and skip PPO RL")
    bench.add_argument("--llm-model", default=None, help="Override LLM model (e.g., gpt-5)")

    return parser


def _handle_profile(args: argparse.Namespace) -> int:
    """Execute the ``profile`` command.

    Returns
    -------
    int
        ``0`` on success, non-zero on failure.
    """

    # If a lightweight profiler is injected (tests), use it to avoid heavy work
    _pm = globals().get("profile_model")
    if callable(_pm):  # type: ignore[call-arg]
        try:
            import torch  # lazy import
            if not args.model.exists():
                print(f"Model file {args.model} not found.")
                return 1
            model = torch.jit.load(str(args.model))
            shape = _parse_shape(args.input_shape)
            x = torch.randn(*shape)
            _pm(model, x, str(args.output))  # type: ignore[misc]
            print(f"Telemetry written: {args.output}")
            return 0
        except Exception as exc:
            print(f"Lightweight profiling failed: {exc}")
            return 1

    # Delegate to the richer CLI profiler to produce full artifacts and progress
    try:
        from . import profile as _prof
    except Exception as exc:
        print(f"Failed to import profiler: {exc}")
        return 1

    artifacts_dir = _Path("agnitraai") / "context"
    return _prof.run(
        model_path=args.model,
        input_shape=_parse_shape(args.input_shape),
        telemetry_out=_Path(args.output),
        artifacts_dir=artifacts_dir,
    )


def _handle_benchmark(args: argparse.Namespace) -> int:
    try:
        import torch  # lazy import
    except Exception:
        print("PyTorch is required for benchmarking but is not installed.")
        return 1

    if not args.model.exists():
        print(f"Model file {args.model} not found.")
        return 1
    try:
        model = torch.jit.load(str(args.model))
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return 1

    model.eval()
    shape = _parse_shape(args.input_shape)
    input_tensor = torch.randn(*shape)

    # propagate feature flags conveniently from CLI args
    import os
    if args.llm_rl:
        os.environ["AGNITRA_ENABLE_LLM_RL"] = "1"
    if args.only_llm:
        os.environ["AGNITRA_ONLY_LLM"] = "1"
    if args.llm_model:
        os.environ["AGNITRA_LLM_MODEL"] = args.llm_model

    from agnitra.benchmarks import run_benchmark

    try:
        summary = run_benchmark(
            model,
            input_tensor,
            out_dir=args.output_dir,
            repeats=args.repeats,
            warmup=args.warmup,
            enable_rl=not args.only_llm,
        )
    except Exception as exc:
        print(f"Benchmark failed: {exc}")
        return 1

    print(
        f"Speedup: {summary['speedup']:.3f}x | latency {summary['latency_before_ms']:.3f} -> {summary['latency_after_ms']:.3f} ms"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Agnitra CLI.

    Parameters
    ----------
    argv:
        Optional list of arguments. When ``None``, ``sys.argv`` is used.

    Returns
    -------
    int
        Exit code where ``0`` indicates success.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "profile":
        return _handle_profile(args)
    if args.cmd == "benchmark":
        return _handle_benchmark(args)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
