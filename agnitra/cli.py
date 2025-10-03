"""Click-based CLI for Agnitra."""
from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, TYPE_CHECKING

import click

from . import optimize_model
from .sdk import resolve_input_tensor

if TYPE_CHECKING:  # pragma: no cover - help type checkers only
    import torch
    from torch import nn, Tensor

_LOG = logging.getLogger(__name__)


def _parse_shape(_: click.Context, __: click.Parameter, value: Optional[str]) -> Optional[Sequence[int]]:
    if value is None:
        return None
    try:
        return tuple(int(dim) for dim in value.split(",") if dim)
    except ValueError as exc:  # pragma: no cover - user input error
        raise click.BadParameter("input shape must be comma separated integers") from exc


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(package_name="agnitra")
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

    torch, _ = _require_torch()

    try:
        model = _load_model(model_path, torch)
    except Exception as exc:  # pragma: no cover - load errors covered via CLI tests
        raise click.ClickException(f"Failed to load model: {exc}") from exc

    device = None
    if device_name:
        try:
            device = torch.device(device_name)
        except Exception as exc:
            raise click.ClickException(f"Invalid device '{device_name}': {exc}") from exc
        if hasattr(model, "to"):
            try:
                model = model.to(device)  # type: ignore[assignment]
            except Exception as exc:
                raise click.ClickException(f"Unable to move model to {device_name}: {exc}") from exc

    try:
        sample = resolve_input_tensor(model, input_tensor=None, input_shape=input_shape, device=device)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    try:
        optimized = optimize_model(
            model,
            input_tensor=sample,
            input_shape=None,
            device=device,
            enable_rl=not disable_rl,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    output_path = _resolve_output_path(model_path, output_path)

    try:
        _save_model(optimized, output_path, sample, torch)
    except Exception as exc:
        raise click.ClickException(f"Failed to save optimized model: {exc}") from exc

    click.echo(f"Optimized model written to {output_path}")


def _load_model(model_path: Path, torch_mod: ModuleType) -> "nn.Module":
    if model_path.suffix in {".pt", ".pth"}:
        # Prefer TorchScript load but fall back to torch.load when necessary.
        try:
            return torch_mod.jit.load(str(model_path))
        except Exception:
            _LOG.debug("torch.jit.load failed; falling back to torch.load", exc_info=True)
            return torch_mod.load(str(model_path), map_location="cpu")

    raise ValueError(f"Unsupported model format: {model_path.suffix}")


def _resolve_output_path(model_path: Path, output_path: Optional[Path]) -> Path:
    if output_path is not None:
        return output_path
    suffix = model_path.suffix or ".pt"
    return model_path.with_name(f"{model_path.stem}_optimized{suffix}")


def _save_model(
    model: "nn.Module",
    output_path: Path,
    sample: "Tensor",
    torch_mod: ModuleType,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save"):
        model.save(str(output_path))  # type: ignore[attr-defined]
        return

    tensor = _align_tensor_with_module(sample, model, torch_mod)
    with torch_mod.inference_mode():
        scripted = torch_mod.jit.trace(model, tensor)
    scripted.save(str(output_path))


def _align_tensor_with_module(sample: "Tensor", model: "nn.Module", torch_mod: ModuleType) -> "Tensor":
    tensor = sample.detach()
    target = _infer_module_device(model, torch_mod)
    if target is None:
        return tensor.cpu()
    return tensor.to(target)


def _infer_module_device(model: "nn.Module", torch_mod: ModuleType) -> Optional["torch.device"]:
    for accessor in ("parameters", "buffers"):
        if not hasattr(model, accessor):
            continue
        try:
            items = getattr(model, accessor)()  # type: ignore[call-arg]
        except Exception:
            continue
        for item in items:
            if isinstance(item, torch_mod.Tensor):
                return item.device
    return None


def _require_torch() -> tuple[ModuleType, ModuleType]:
    try:
        torch_mod = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - torch absent
        raise click.ClickException("PyTorch is required to run optimization.") from exc

    nn_mod = getattr(torch_mod, "nn", None)
    if nn_mod is None:  # pragma: no cover - very unlikely
        raise click.ClickException("torch.nn is unavailable in this environment")

    return torch_mod, nn_mod


def main() -> None:
    """Console script entry point."""

    cli(prog_name="agnitra")


if __name__ == "__main__":  # pragma: no cover
    main()
