#!/usr/bin/env bash
# One-command repro for the Llama-3-8B / H100 benchmark.
#
# A failure in any single runner does NOT abort the others — the goal is
# to always produce a complete RESULTS.md, with missing rows visible.
#
# Usage:
#   HF_TOKEN=hf_xxx ./run.sh           # standard run
#   ./run.sh --clean                   # wipe raw/ first (rerun from scratch)
#   AGNITRA_BENCH_GPU=2 ./run.sh       # pin to a specific GPU index
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    -h|--help)
      sed -n '1,12p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# --- Preflight: GPU pin -----------------------------------------------------
# Pin to a single GPU so multi-GPU machines produce a single-card number,
# never a partially-parallel one. Override with AGNITRA_BENCH_GPU.
export CUDA_VISIBLE_DEVICES="${AGNITRA_BENCH_GPU:-0}"

# --- Preflight: HF_TOKEN ----------------------------------------------------
if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  cat <<'EOF' >&2
ERROR: HF_TOKEN is not set.

Llama-3-8B-Instruct is a gated model. Each runner will fail with a 401
without an authorized token. Request access at
https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct and then:

    export HF_TOKEN=hf_xxx
    ./run.sh

To run anyway (e.g. you've prefetched weights into HF_HOME), set
AGNITRA_BENCH_SKIP_HF_CHECK=1.
EOF
  if [ "${AGNITRA_BENCH_SKIP_HF_CHECK:-0}" != "1" ]; then
    exit 3
  fi
fi
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"

# --- Preflight: GPU presence and SKU ----------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "WARN: nvidia-smi not found — runners will write JSON with success=false." >&2
fi

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1 || echo unknown)"
case "$GPU_NAME" in
  *H100*) ;;
  unknown) echo "WARN: could not detect GPU name." >&2 ;;
  *)
    cat >&2 <<EOF
WARN: detected GPU '$GPU_NAME', not an H100.

This benchmark suite is named llama3_h100 because the published numbers
are H100. The runner code will still execute, but DO NOT compare your
output to the H100 baseline — it is not a like-for-like measurement.
Continuing in 5s...
EOF
    sleep 5
    ;;
esac

# --- Optional clean ---------------------------------------------------------
RAW="$HERE/raw"
if [ "$CLEAN" = "1" ]; then
  echo "removing $RAW (--clean)"
  rm -rf "$RAW"
fi
mkdir -p "$RAW"

# --- 1) Capture environment so reviewers can see what produced the numbers --
{
  echo "=== nvidia-smi ==="
  nvidia-smi || echo "(nvidia-smi unavailable)"
  echo
  echo "=== CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES ==="
  echo
  echo "=== python ==="
  python --version || python3 --version
  echo
  echo "=== pip freeze ==="
  pip freeze
  echo
  echo "=== git ==="
  (cd "$HERE/../.." && git rev-parse HEAD 2>/dev/null) || echo "(not a git repo)"
} > "$RAW/env.txt" 2>&1

# Pick whichever python interpreter is on PATH; some images only expose python3.
PY="$(command -v python || command -v python3)"
if [ -z "$PY" ]; then
  echo "ERROR: no python interpreter on PATH" >&2
  exit 4
fi

# --- 2) Run each runner. Failures do NOT abort the others -------------------
run_runner () {
  local name="$1"
  local script="$2"
  shift 2
  echo "--- running: $name ---"
  if "$PY" "$script" --output "$RAW/$name.json" "$@"; then
    echo "--- $name: ok ---"
  else
    echo "--- $name: FAILED (continuing) ---"
  fi
}

run_runner "hf"             "runners/hf.py"
run_runner "torch_compile"  "runners/torch_compile.py"
run_runner "vllm"           "runners/vllm_runner.py"
run_runner "tensorrt_llm"   "runners/tensorrt_llm.py"
run_runner "agnitra"        "runners/agnitra_runner.py"

# --- 3) Aggregate -----------------------------------------------------------
"$PY" compare.py --raw "$RAW" --out "$HERE/RESULTS.md"

echo
echo "Done. See $HERE/RESULTS.md and $RAW/."
