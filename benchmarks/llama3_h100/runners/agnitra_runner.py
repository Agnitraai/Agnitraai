"""Agnitra runner — the SUT (system under test).

Calls `agnitra.optimize(model, ...)` to produce an optimized model and
then runs the standard HF-style measurement loop. This must use the
public SDK entry point — never private internals — so the benchmark
matches what a user would actually run.
"""
from __future__ import annotations

import os
import sys

from _hf_base import parse_args, run_hf_style
from workload import WORKLOAD


def _optimize(model):
    import torch
    import agnitra

    # Synthesize an example input matching the workload's input_tokens
    # so Agnitra's profiler sees realistic shapes. Greedy decode at
    # input_len=512, batch=1 is the canonical fingerprint here; the
    # optimizer caches per-fingerprint, so subsequent batch sizes
    # benefit from the same cached profile.
    example = torch.zeros(
        (1, WORKLOAD.input_tokens),
        dtype=torch.long,
        device=next(model.parameters()).device,
    )

    result = agnitra.optimize(
        model,
        input_tensor=example,
        project_id="benchmark/llama3_h100",
        model_name=WORKLOAD.model_id,
        # Disable RL for benchmark reproducibility — RL introduces
        # variance run-to-run that makes regressions hard to detect.
        # Run the RL variant separately if you want to measure it.
        enable_rl=False,
        offline=True,
        repeats=3,
        warmup=1,
    )
    return result.optimized_model


def main() -> int:
    args = parse_args(default_runner="agnitra")
    try:
        import agnitra
        version = f"agnitra {getattr(agnitra, '__version__', 'unknown')}"
    except Exception:
        version = "agnitra (unknown)"

    return run_hf_style(
        runner_name=args.runner_name,
        library_version=version,
        prepare_model=_optimize,
        output_path=args.output,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        extra={"enable_rl": False, "offline": True},
    )


if __name__ == "__main__":
    sys.exit(main())
