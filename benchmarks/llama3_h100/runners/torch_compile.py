"""`torch.compile` baseline — what every PyTorch user gets for free.

If Agnitra cannot beat `torch.compile(mode="reduce-overhead")`, the
optimizer's value proposition is suspect. This runner exists to make
that comparison unavoidable.
"""
from __future__ import annotations

import os
import sys

from _hf_base import parse_args, run_hf_style


def _compile(model):
    import torch

    # `reduce-overhead` reuses CUDA graphs across calls — generally the
    # best mode for autoregressive decode at fixed shapes. We do NOT
    # toggle `mode="max-autotune"` because the long compile time hurts
    # CI feedback and the gains are usually small for decode.
    return torch.compile(model, mode="reduce-overhead", fullgraph=False)


def main() -> int:
    args = parse_args(default_runner="torch_compile")
    try:
        import torch
        version = f"torch.compile (torch {torch.__version__})"
    except Exception:
        version = "torch.compile (unknown)"

    return run_hf_style(
        runner_name=args.runner_name,
        library_version=version,
        prepare_model=_compile,
        output_path=args.output,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        extra={"torch_compile_mode": "reduce-overhead"},
    )


if __name__ == "__main__":
    sys.exit(main())
