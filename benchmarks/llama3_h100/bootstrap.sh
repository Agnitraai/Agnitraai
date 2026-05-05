#!/usr/bin/env bash
# Bootstrap a fresh Linux box (Lambda Labs / RunPod / vast.ai / bare metal)
# with everything `run.sh` needs.
#
# Idempotent: safe to re-run. Skips work already done.
#
# Tested on Ubuntu 22.04 LTS with NVIDIA driver 535+ and CUDA 12.1
# already installed (which is the default on Lambda's H100 images).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
cd "$REPO_ROOT"

echo "==> Bootstrap from $REPO_ROOT"

# 1. nvidia-smi has to work. If it doesn't, the box has no driver and
#    none of this matters.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi missing. Install the NVIDIA driver before running this." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# 2. Python 3.11. Ubuntu 22.04 ships 3.10 by default; we need 3.11
#    because the pinned torch/vllm wheels target it.
if ! command -v python3.11 >/dev/null 2>&1; then
  echo "==> Installing Python 3.11"
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
  else
    echo "ERROR: no apt-get; install Python 3.11 manually." >&2
    exit 1
  fi
fi

# 3. Virtual environment.
VENV="$REPO_ROOT/.venv-bench"
if [ ! -d "$VENV" ]; then
  echo "==> Creating venv at $VENV"
  python3.11 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel

# 4. Pinned benchmark deps + Agnitra (editable).
echo "==> Installing benchmark requirements"
pip install -r "$HERE/requirements.txt"
echo "==> Installing Agnitra from source"
pip install -e "$REPO_ROOT" --no-deps
# Faster HF downloads.
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# 5. Sanity print.
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda?", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

cat <<EOF

==> Bootstrap complete.

To run the benchmark:

    source $VENV/bin/activate
    export HF_TOKEN=hf_xxx
    cd $HERE
    ./run.sh

EOF
