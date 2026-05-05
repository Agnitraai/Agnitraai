"""HuggingFace `transformers` baseline — `model.generate()` with no tricks.

This is the floor: what someone gets from a one-line HuggingFace install
without any optimization layer. Other runners must beat this number to
have a story.
"""
from __future__ import annotations

import os
import sys

from _hf_base import parse_args, run_hf_style


def main() -> int:
    args = parse_args(default_runner="hf")
    try:
        import transformers
        version = f"transformers {transformers.__version__}"
    except Exception:
        version = "transformers (unknown)"

    return run_hf_style(
        runner_name=args.runner_name,
        library_version=version,
        prepare_model=lambda model: model,
        output_path=args.output,
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
    )


if __name__ == "__main__":
    sys.exit(main())
