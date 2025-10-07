"""Milestone demo showcasing Agnitra's optimization workflow.

This script illustrates three facets of the platform:

1. Baseline vs optimized latency using the in-repo TinyLlama fixture.
2. CLI usage via the Click-powered ``agnitra optimize`` command.
3. Kernel injection using the runtime patcher to swap an FX node.

Run ``python demo.py`` from the repository root. Use ``--help`` to
inspect additional knobs (e.g., sample size, repeats).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

from agnitra.core.kernel import KernelGenerator
from agnitra.core.metering import UsageMeter
from agnitra.core.runtime import (
    FXNodePatch,
    ForwardHookPatch,
    RuntimeOptimizationAgent,
    RuntimePatcher,
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tinyllama.pt"


def _ensure_model(verbose: bool = True) -> Path:
    """Create the TinyLlama TorchScript artifact when absent."""

    if MODEL_PATH.exists():
        if verbose:
            print(f"[model] Reusing existing {MODEL_PATH.relative_to(ROOT)}")
        return MODEL_PATH

    if verbose:
        print("[model] tinyllama.pt missing – generating via prepare_tinyllama.py")
    subprocess.run([sys.executable, "prepare_tinyllama.py"], check=True, cwd=str(ROOT))
    if not MODEL_PATH.exists():  # pragma: no cover - defensive guard
        raise FileNotFoundError("Expected tinyllama.pt after running prepare_tinyllama.py")
    return MODEL_PATH


def _pretty_pct(delta: float) -> str:
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def _infer_device(module: torch.nn.Module) -> Optional[torch.device]:
    for accessor in ("parameters", "buffers"):
        if not hasattr(module, accessor):
            continue
        try:
            iterator = getattr(module, accessor)()  # type: ignore[call-arg]
        except Exception:
            continue
        for item in iterator:
            if isinstance(item, torch.Tensor):
                return item.device
    return None


def _align_for_module(module: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    target = _infer_device(module)
    if target is not None and tensor.device != target:
        return tensor.to(target)
    return tensor


def demonstrate_baseline_vs_optimized(sample_shape: Tuple[int, ...], *, repeats: int, warmup: int) -> None:
    model_path = _ensure_model()
    print("\n=== Baseline vs optimized (Python SDK) ===")

    os.environ.setdefault("KINETO_LOG_LEVEL", "5")
    logging.getLogger("agnitra._sdk.optimizer").setLevel(logging.CRITICAL)

    module = torch.jit.load(str(model_path)).eval()
    sample = torch.randn(*sample_shape)

    meter = UsageMeter(rate_per_gpu_hour=3.0, margin_pct=0.25)
    agent = RuntimeOptimizationAgent(
        usage_meter=meter,
        repeats=repeats,
        warmup=warmup,
        rate_per_gpu_hour=3.0,
        success_margin_pct=0.25,
    )
    result = agent.optimize(
        module,
        sample,
        project_id="demo",
        model_name="TinyLlama",
        enable_rl=False,
        metadata={"source": "demo.baseline"},
    )

    optimized_module = result.optimized_model
    baseline_snapshot = result.baseline
    optimized_snapshot = result.optimized
    usage_event = result.usage_event

    print(f"Baseline latency : {baseline_snapshot.latency_ms:.3f} ms ({baseline_snapshot.tokens_per_sec:.1f} tokens/s)")
    print(f"Optimized latency: {optimized_snapshot.latency_ms:.3f} ms ({optimized_snapshot.tokens_per_sec:.1f} tokens/s)")

    if usage_event is not None:
        print(
            "Improvement      : "
            f"{usage_event.performance_uplift_pct:.1f}% uplift"
        )
        print(
            "Billing snapshot : "
            f"GPU hours saved {usage_event.gpu_hours_saved:.6f}, "
            f"billable {usage_event.total_billable:.4f} {usage_event.currency}"
        )
    else:
        uplift_pct = (
            (baseline_snapshot.latency_ms - optimized_snapshot.latency_ms) / baseline_snapshot.latency_ms * 100
            if baseline_snapshot.latency_ms
            else 0.0
        )
        print(f"Improvement      : {_pretty_pct(uplift_pct)}")

    baseline_util = baseline_snapshot.gpu_utilization
    optimized_util = optimized_snapshot.gpu_utilization
    if baseline_util is not None or optimized_util is not None:
        before = f"{baseline_util:.1f}%" if baseline_util is not None else "n/a"
        after = f"{optimized_util:.1f}%" if optimized_util is not None else "n/a"
        print(f"GPU utilisation  : {before} -> {after}")

    with torch.inference_mode():
        sample_aligned = _align_for_module(optimized_module, sample.clone())
        baseline_out = module(_align_for_module(module, sample.clone()))
        optimized_out = optimized_module(sample_aligned)
        delta = torch.max(torch.abs(optimized_out - baseline_out))
    print(f"L_inf difference  : {float(delta):.3e}\n")


def demonstrate_cli(sample_shape: Tuple[int, ...]) -> None:
    print("\n=== CLI optimization ===")
    model_path = _ensure_model(verbose=False)
    shape_literal = ",".join(str(dim) for dim in sample_shape)
    output_path = model_path.with_name(f"{model_path.stem}_cli.pt")

    cmd = ["agnitra", "optimize", "--model", str(model_path), "--input-shape", shape_literal, "--output", str(output_path)]
    if not shutil.which("agnitra"):
        cmd = [sys.executable, "-m", "agnitra.cli", "optimize", "--model", str(model_path), "--input-shape", shape_literal, "--output", str(output_path)]

    env = os.environ.copy()
    env.setdefault("KINETO_LOG_LEVEL", "5")
    completed = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), check=True, env=env)
    print(completed.stdout.strip() or completed.stderr.strip())
    if output_path.exists():
        print(f"Output artifact available at {output_path.relative_to(ROOT)}\n")


def demonstrate_kernel_injection() -> None:
    print("\n=== Kernel injection demo ===")
    generator = KernelGenerator()
    result = generator.generate("runtime-demo-policy", validate=False)

    module_path = result.module_path
    namespace = {}
    exec(compile(module_path.read_text(), str(module_path), "exec"), namespace)  # noqa: S102 - deliberate exec for demo
    run_kernel = namespace.get("run_kernel")
    if run_kernel is None:
        raise RuntimeError("Generated kernel missing run_kernel")

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.relu = torch.nn.ReLU()

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.relu(x + y)

    patcher = RuntimePatcher()
    fx_patch = FXNodePatch(name="vector-add", target="operator.add", kernel=run_kernel, metadata={"policy": "runtime-demo"})
    hook_patch = ForwardHookPatch(name="relu-scale", module_path="relu", kernel=lambda _mod, _inputs, output: output * 0.5)

    x = torch.arange(8, dtype=torch.float32)
    y = torch.linspace(0.2, 1.0, steps=8)

    model = Toy()
    baseline = model(x, y)
    report = patcher.patch(model, fx_patches=[fx_patch], hook_patches=[hook_patch], copy_module=True)
    optimized = report.module(x, y)

    print("Kernel info      :", module_path.name)
    print("Baseline sum     :", float(baseline.sum()))
    print("Patched sum      :", float(optimized.sum()))
    print("Runtime patches  :", json.dumps([log.status for log in report.logs]))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the milestone demo")
    parser.add_argument("--sample-shape", default="1,16,64", help="Input tensor shape for the TinyLlama demo")
    parser.add_argument("--repeats", type=int, default=10, help="Timed iterations per latency measurement")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per measurement")
    parser.add_argument("--skip-cli", action="store_true", help="Skip invoking the CLI optimize command")
    parser.add_argument("--skip-kernel", action="store_true", help="Skip the runtime kernel injection demo")
    return parser.parse_args(argv)


def _parse_shape_literal(literal: str) -> Tuple[int, ...]:
    try:
        values = tuple(int(dim) for dim in literal.split(",") if dim)
        if not values:
            raise ValueError
        return values
    except ValueError as exc:
        raise SystemExit(f"Invalid --sample-shape '{literal}'. Use comma separated ints (e.g., 1,16,64).") from exc


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    sample_shape = _parse_shape_literal(args.sample_shape)

    demonstrate_baseline_vs_optimized(sample_shape, repeats=args.repeats, warmup=args.warmup)
    if not args.skip_cli:
        demonstrate_cli(sample_shape)
    if not args.skip_kernel:
        demonstrate_kernel_injection()


if __name__ == "__main__":  # pragma: no cover
    main()
