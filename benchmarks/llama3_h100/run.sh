#!/usr/bin/env bash
# One-command repro for the Llama-3-8B / H100 benchmark.
#
# A failure in any single runner does NOT abort the others — the goal is
# to always produce a complete RESULTS.md, with missing rows visible.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

RAW="$HERE/raw"
mkdir -p "$RAW"

# 1) Capture environment so reviewers can see exactly what produced the numbers.
{
  echo "=== nvidia-smi ==="
  nvidia-smi || echo "(nvidia-smi unavailable)"
  echo
  echo "=== python ==="
  python --version
  echo
  echo "=== pip freeze ==="
  pip freeze
  echo
  echo "=== git ==="
  (cd "$HERE/../.." && git rev-parse HEAD 2>/dev/null) || echo "(not a git repo)"
} > "$RAW/env.txt" 2>&1

# 2) Run each runner. They write their own JSON; we don't fail the whole
#    script if one runner can't initialize.
run_runner () {
  local name="$1"
  local script="$2"
  shift 2
  echo "--- running: $name ---"
  if python "$script" --output "$RAW/$name.json" "$@"; then
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

# 3) Aggregate.
python compare.py --raw "$RAW" --out "$HERE/RESULTS.md"

echo
echo "Done. See $HERE/RESULTS.md and $RAW/."
