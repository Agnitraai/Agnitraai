"""Run the Llama-3-8B / H100 benchmark on Modal.

Usage:

    pip install modal
    modal token new                       # one-time auth
    HF_TOKEN=hf_xxx modal run benchmarks/llama3_h100/modal_runner.py

The job:
  1. Builds a Modal image equivalent to the Dockerfile (CUDA 12.1, the
     pinned requirements.txt, Agnitra installed from source).
  2. Provisions a single H100 80GB.
  3. Runs ``./run.sh`` inside the container.
  4. Streams logs to your terminal.
  5. Downloads ``raw/*.json`` and ``RESULTS.md`` back to your local
     ``benchmarks/llama3_h100/`` directory so you can commit them.

Cost (as of 2025): a full benchmark run is roughly 30–45 minutes of
single-H100 wall time, ~$3–5 at Modal's published rate. You can
shorten the run by exporting AGNITRA_BENCH_BATCHES="1" before invoking,
which trims the suite to batch_size=1 only (~10 minutes).
"""
from __future__ import annotations

import os
import pathlib
import sys

try:
    import modal  # type: ignore
except ImportError:  # pragma: no cover - friendly error for local users
    sys.stderr.write(
        "modal package not installed. Run `pip install modal` first.\n"
    )
    sys.exit(2)


# When this file runs locally (image build, local_entrypoint), __file__
# points at the path inside the repo. When Modal imports it inside the
# container, __file__ is /root/modal_runner.py — only one parent level
# — so parents[2] would IndexError. Fall back to the in-container paths
# that match where ``add_local_dir`` mounts the repo below.
def _resolve_paths() -> tuple[pathlib.Path, pathlib.Path]:
    here = pathlib.Path(__file__).resolve()
    try:
        return here.parents[2], here.parent
    except IndexError:
        return pathlib.Path("/work"), pathlib.Path("/work/benchmarks/llama3_h100")


REPO_ROOT, BENCH_DIR = _resolve_paths()

app = modal.App("agnitra-bench-llama3-h100")

# Mirror the Dockerfile exactly. If the Dockerfile changes, change this
# image spec in the same commit so the two stay in sync.
image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        add_python="3.11",
    )
    .apt_install("git", "curl", "ca-certificates", "build-essential")
    .pip_install_from_requirements(
        str(BENCH_DIR / "requirements.txt"),
    )
    .pip_install("hf_transfer")
    # Bake the repo into the image so the container has Agnitra plus
    # the benchmark scripts. ``copy=True`` makes this a build-time copy
    # (cached image layer), so subsequent runs reuse it. Ignore patterns
    # keep the upload small — `.git`, notebooks, demo artifacts, and
    # local virtualenvs balloon image size for no benefit.
    .add_local_dir(
        str(REPO_ROOT),
        remote_path="/work",
        copy=True,
        ignore=[
            ".git",
            ".venv*",
            "**/__pycache__",
            "**/*.ipynb",
            "demo_artifacts",
            "reports",
            "samples",
            "context",
            "internal-docs",
            "agnitra.egg-info",
            "telemetry*.json",
            "**/*.pt",
            "**/*.pth",
        ],
    )
    .run_commands(
        "cd /work && pip install -e . --no-deps",
    )
    .env({
        "PYTHONUNBUFFERED": "1",
        # hf_transfer (Rust-accelerated download) hits "no permits
        # available" connection-pool errors on Modal's network when
        # downloading multi-shard Llama-3 weights. Disable; fall back
        # to the standard requests-based downloader. Slower (~minutes)
        # but reliable.
        "HF_HUB_ENABLE_HF_TRANSFER": "0",
        "PIP_NO_CACHE_DIR": "1",
    })
)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,  # 1 hour — covers slow first-time HF download.
    secrets=[modal.Secret.from_dict({
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "HUGGING_FACE_HUB_TOKEN": os.environ.get("HF_TOKEN", ""),
    })],
)
def run_benchmark() -> dict[str, bytes]:
    """Run ./run.sh and return raw/*.json + RESULTS.md as bytes."""
    import os
    import subprocess

    work_dir = "/work/benchmarks/llama3_h100"
    os.chdir(work_dir)
    # `--clean` to avoid carryover between Modal invocations.
    subprocess.run(["./run.sh", "--clean"], check=False)

    artifacts: dict[str, bytes] = {}
    raw_dir = os.path.join(work_dir, "raw")
    if os.path.isdir(raw_dir):
        for name in sorted(os.listdir(raw_dir)):
            full = os.path.join(raw_dir, name)
            if os.path.isfile(full):
                with open(full, "rb") as handle:
                    artifacts[f"raw/{name}"] = handle.read()
    results = os.path.join(work_dir, "RESULTS.md")
    if os.path.isfile(results):
        with open(results, "rb") as handle:
            artifacts["RESULTS.md"] = handle.read()
    return artifacts


@app.local_entrypoint()
def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        sys.stderr.write(
            "ERROR: HF_TOKEN is not set in your local shell. Llama-3 is gated;\n"
            "Modal cannot download the model without a token.\n"
            "  export HF_TOKEN=hf_xxx && modal run modal_runner.py\n"
        )
        sys.exit(3)

    print("Launching benchmark on a Modal H100. First run downloads weights "
          "(~16 GB) and may take several minutes.")
    artifacts = run_benchmark.remote()

    out_dir = BENCH_DIR
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for relpath, contents in artifacts.items():
        target = out_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        print(f"  wrote {target.relative_to(REPO_ROOT)}")

    print("\nDone. Review benchmarks/llama3_h100/RESULTS.md and commit.")
