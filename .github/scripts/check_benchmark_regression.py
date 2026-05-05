"""Fail CI if Agnitra's batch=1 throughput regressed >threshold vs. baseline.

Compares the freshly produced ``raw/agnitra.json`` against the version
checked in at the baseline ref (default: ``origin/main``). If the
baseline is missing (first-ever run, or rename), the script logs a
warning and exits zero — so the gate never blocks the very first
publication of the benchmark.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def _baseline_json(ref: str, repo_path: str) -> dict | None:
    try:
        out = subprocess.check_output(
            ["git", "show", f"{ref}:{repo_path}"],
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return None
    return json.loads(out.decode())


def _batch_throughput(blob: dict, batch_size: int) -> float | None:
    for batch in blob.get("batches", []):
        if batch["batch_size"] == batch_size:
            return batch.get("throughput_tps")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--baseline-ref", default="origin/main")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    current_path = args.raw / "agnitra.json"
    if not current_path.is_file():
        print(f"ERROR: {current_path} not found — Agnitra runner did not produce results.")
        return 1

    current = _load(current_path)
    if not current.get("success"):
        print(f"ERROR: Agnitra runner reported failure: {current.get('error', 'unknown')}")
        return 1

    cur_tps = _batch_throughput(current, args.batch_size)
    if cur_tps is None or cur_tps <= 0:
        print(f"ERROR: no throughput recorded at batch={args.batch_size}.")
        return 1

    baseline = _baseline_json(
        args.baseline_ref, str(current_path).removeprefix("./")
    )
    if baseline is None:
        print(
            f"WARN: no baseline at {args.baseline_ref}:{current_path} — "
            "skipping regression check."
        )
        return 0
    if not baseline.get("success"):
        print("WARN: baseline marked unsuccessful — skipping regression check.")
        return 0

    base_tps = _batch_throughput(baseline, args.batch_size)
    if base_tps is None or base_tps <= 0:
        print("WARN: baseline missing throughput — skipping regression check.")
        return 0

    delta = (cur_tps - base_tps) / base_tps
    print(
        f"throughput at batch={args.batch_size}: "
        f"baseline={base_tps:.1f} tok/s, current={cur_tps:.1f} tok/s, "
        f"delta={delta * 100:+.2f}%"
    )

    if delta < -args.threshold:
        print(
            f"FAIL: regression {delta * 100:.2f}% exceeds threshold "
            f"-{args.threshold * 100:.2f}%."
        )
        return 1

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
