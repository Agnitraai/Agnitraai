"""Standalone benchmark runner for Agnitra.

This script loads a TorchScript model, runs both baseline and optimized
inference using :func:`agnitra.benchmarks.run_benchmark`, and persists a set of
artifacts:

* ``before.json`` / ``after.json`` containing raw measurement payloads.
* ``summary.json`` and ``summary_diff.json`` with aggregated metrics.
* ``summary.csv`` collecting the most relevant comparisons in tabular form.
* ``benchmark_plots.png`` visualising latency and throughput side-by-side.

The script is intentionally thin so it can be orchestrated by higher level
automation (e.g. notebooks or CI pipelines) while also being directly
invokable from the command line.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

from agnitra.benchmarks import run_benchmark

LOGGER = logging.getLogger("agnitra.benchmark_runner")


def _parse_shape(raw: str) -> Sequence[int]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _load_torch() -> Any:
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required to execute the benchmark runner") from exc


def _load_model(model_path: Path) -> Any:
    torch = _load_torch()
    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found")

    if model_path.suffix in {".pt", ".pth", ".ts"}:
        return torch.jit.load(str(model_path))

    raise ValueError(
        "Only TorchScript (.pt/.pth/.ts) models are supported. Provide a compiled artifact."
    )


def _build_input(shape: Sequence[int]) -> Any:
    torch = _load_torch()
    if not shape:
        raise ValueError("Input shape must contain at least one dimension")
    return torch.randn(*shape)


def _write_csv(summary: Dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    baseline = summary.get("results", {}).get("baseline", {})
    optimized = summary.get("results", {}).get("optimized", {})

    rows = [
        {
            "variant": "baseline",
            "latency_ms": summary.get("latency_before_ms", 0.0),
            "memory_gb": summary.get("memory_before_gb", 0.0),
            "tokens_per_sec": summary.get("tokens_before_per_sec", 0.0),
        },
        {
            "variant": "optimized",
            "latency_ms": summary.get("latency_after_ms", 0.0),
            "memory_gb": summary.get("memory_after_gb", 0.0),
            "tokens_per_sec": summary.get("tokens_after_per_sec", 0.0),
        },
    ]

    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (destination.parent / "summary_details.json").open("w", encoding="utf-8") as fh:
        json.dump({"baseline": baseline, "optimized": optimized}, fh, indent=2)


def _create_plots(summary: Dict[str, Any], destination: Path) -> None:
    try:  # pragma: no cover - matplotlib is optional in minimal test envs
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOGGER.info("Matplotlib unavailable; skipping plot generation: %s", exc)
        return

    labels = ["baseline", "optimized"]
    latency = [summary.get("latency_before_ms", 0.0), summary.get("latency_after_ms", 0.0)]
    tokens = [summary.get("tokens_before_per_sec", 0.0), summary.get("tokens_after_per_sec", 0.0)]

    figure = plt.figure(figsize=(8, 4))
    ax1 = figure.add_subplot(1, 2, 1)
    ax1.bar(labels, latency, color=["#8888ff", "#55aa55"])
    ax1.set_title("Latency (ms)")
    ax1.set_ylabel("Average per inference")

    ax2 = figure.add_subplot(1, 2, 2)
    ax2.bar(labels, tokens, color=["#ffa500", "#1f77b4"])
    ax2.set_title("Tokens / sec")

    figure.suptitle("Agnitra Benchmark Comparison")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def run_single_benchmark(
    model: Any,
    input_tensor: Any,
    out_dir: Path,
    *,
    repeats: int,
    warmup: int,
    enable_rl: bool,
    token_count: int | None = None,
) -> Dict[str, Any]:
    summary = run_benchmark(
        model,
        input_tensor,
        out_dir=out_dir,
        repeats=repeats,
        warmup=warmup,
        enable_rl=enable_rl,
        token_count=token_count,
    )

    _write_csv(summary, out_dir / "summary.csv")
    _create_plots(summary, out_dir / "benchmark_plots.png")
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Agnitra benchmark pipeline")
    parser.add_argument("model", type=Path, help="Path to TorchScript model (.pt/.pth/.ts)")
    parser.add_argument(
        "--input-shape",
        default="1,16",
        help="Comma separated dimensions for synthetic input tensor (default: 1,16)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks",
        help="Directory where benchmark artifacts will be written",
    )
    parser.add_argument("--repeats", type=int, default=10, help="Number of recorded iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup runs before timing")
    parser.add_argument(
        "--enable-rl",
        action="store_true",
        help="Enable PPO/Codex tuning pipeline when optimising the model",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    try:
        shape = _parse_shape(args.input_shape)
        model = _load_model(Path(args.model))
        input_tensor = _build_input(shape)
        out_dir = Path(args.output_dir)
        summary = run_single_benchmark(
            model,
            input_tensor,
            out_dir,
            repeats=args.repeats,
            warmup=args.warmup,
            enable_rl=args.enable_rl,
        )
    except Exception as exc:
        LOGGER.error("Benchmark failed: %s", exc)
        return 1

    speedup = summary.get("speedup", 1.0)
    LOGGER.info("Speedup %.3fx | latency %.3f -> %.3f ms", speedup, summary.get("latency_before_ms", 0.0), summary.get("latency_after_ms", 0.0))
    LOGGER.info("Artifacts written to %s", Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

