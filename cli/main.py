"""Command line interface for Agnitra.

The CLI is designed for resilience: heavy dependencies like ``torch`` are
imported lazily and all commands return an exit code instead of raising
exceptions.  This allows callers to handle errors gracefully and enables
"self-healing" behaviours in higher level orchestrators.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from agnitra.telemetry_collector import profile_model


def _parse_shape(s: str) -> Sequence[int]:
    """Parse a comma separated shape string into a sequence of ints."""

    return tuple(int(x) for x in s.split(","))


def _build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the CLI.

    Splitting parser construction allows tests to exercise argument handling
    without executing command logic.
    """

    parser = argparse.ArgumentParser(prog="agnitra")
    sub = parser.add_subparsers(dest="cmd", required=True)

    prof = sub.add_parser("profile", help="Profile a Torch model")
    prof.add_argument("model", type=Path, help="Path to a TorchScript model")
    prof.add_argument(
        "--input-shape",
        default="1,3,224,224",
        help="Comma separated input tensor shape",
    )
    prof.add_argument(
        "--output",
        default="telemetry.json",
        help="Path to write telemetry JSON",
    )

    return parser


def _handle_profile(args: argparse.Namespace) -> int:
    """Execute the ``profile`` command.

    Returns
    -------
    int
        ``0`` on success, non-zero on failure.
    """

    try:
        import torch  # imported lazily for graceful degradation
    except Exception:  # pragma: no cover - import failure is environment specific
        print("PyTorch is required for profiling but is not installed.")
        return 1

    if not args.model.exists():
        print(f"Model file {args.model} not found.")
        return 1
    try:
        model = torch.jit.load(str(args.model))
    except Exception as exc:  # pragma: no cover - torch specific errors
        print(f"Failed to load model: {exc}")
        return 1

    model.eval()
    shape = _parse_shape(args.input_shape)
    input_tensor = torch.randn(*shape)

    try:
        profile_model(model, input_tensor, args.output)
    except Exception as exc:  # pragma: no cover - profiling failures are rare
        print(f"Profiling failed: {exc}")
        return 1

    print(f"Telemetry written to {args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the Agnitra CLI.

    Parameters
    ----------
    argv:
        Optional list of arguments. When ``None``, ``sys.argv`` is used.

    Returns
    -------
    int
        Exit code where ``0`` indicates success.
    """

    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "profile":
        return _handle_profile(args)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
