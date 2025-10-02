"""Click-based CLI for Agnitra."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import click

try:  # pragma: no cover - optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]

from . import optimize_model

_LOG = logging.getLogger(__name__)


def _parse_shape(_: click.Context, __: click.Parameter, value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    try:
        return tuple(int(dim) for dim in value.split(",") if dim)
    except ValueError as exc:  # pragma: no cover - user input error
        raise click.BadParameter("input shape must be comma separated integers") from exc


@click.group()
def cli() -> None:
    """Agnitra command line interface."""


@cli.command("optimize")
@click.option(
    "model_path",
    "--model",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a TorchScript model (*.pt/*.pth).",
)
@click.option(
    "input_shape",
    "--input-shape",
    callback=_parse_shape,
    default="1,3,224,224",
    show_default=True,
    help="Comma separated tensor shape for a random input sample.",
)
@click.option(
    "output_path",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Destination to save the optimized TorchScript model.",
)
@click.option(
    "device_name",
    "--device",
    default=None,
    help="Execution device (e.g. cpu, cuda). Defaults to model device.",
)
@click.option(
    "disable_rl",
    "--disable-rl",
    is_flag=True,
    help="Disable PPO reinforcement learning stage.",
)
def optimize_command(
    model_path: Path,
    input_shape: Optional[Sequence[int]],
    output_path: Optional[Path],
    device_name: Optional[str],
    disable_rl: bool,
) -> None:
    """Optimize a Torch model and optionally persist the result."""

    if torch is None:  # pragma: no cover - torch absent
        raise click.ClickException("PyTorch is required to run optimization.")

    try:
        model = _load_model(model_path)
    except Exception as exc:  # pragma: no cover - load errors covered via CLI tests
        raise click.ClickException(f"Failed to load model: {exc}") from exc

    if device_name:
        device = torch.device(device_name)
        try:
            model = model.to(device) if hasattr(model, "to") else model
        except Exception as exc:
            raise click.ClickException(f"Unable to move model to {device_name}: {exc}") from exc
    else:
        device = None

    try:
        optimized = optimize_model(
            model,
            input_tensor=None,
            input_shape=input_shape,
            device=device,
            enable_rl=not disable_rl,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = _resolve_output_path(model_path, output_path)

    try:
        _save_model(optimized, output_path, input_shape, device)
    except Exception as exc:
        raise click.ClickException(f"Failed to save optimized model: {exc}") from exc

    click.echo(f"Optimized model written to {output_path}")


def _load_model(model_path: Path) -> "nn.Module":
    if torch is None:  # pragma: no cover - torch absent
        raise RuntimeError("PyTorch is required")

    if model_path.suffix in {".pt", ".pth"}:
        # Prefer TorchScript load but fall back to torch.load when necessary.
        try:
            return torch.jit.load(str(model_path))
        except Exception:
            _LOG.debug("torch.jit.load failed; falling back to torch.load", exc_info=True)
            return torch.load(str(model_path), map_location="cpu")

    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def _resolve_output_path(model_path: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    suffix = model_path.suffix or ".pt"
    return model_path.with_name(f"{model_path.stem}_optimized{suffix}")


def _save_model(
    model: "nn.Module",
    output_path: Path,
    input_shape: Optional[Sequence[int]],
    device: Optional["torch.device"],
) -> None:
    if torch is None:  # pragma: no cover - torch absent
        raise RuntimeError("PyTorch is required")

    if hasattr(model, "save"):
        model.save(str(output_path))  # type: ignore[attr-defined]
        return

    if input_shape is None:
        raise ValueError("input_shape is required when saving traced models")

    sample = torch.randn(*input_shape)
    if device is not None:
        sample = sample.to(device)

    scripted = torch.jit.trace(model, sample)
    scripted.save(str(output_path))


def main() -> None:
    """Console script entry point."""

    cli(prog_name="agnitra")


if __name__ == "__main__":  # pragma: no cover
    main()

