"""Click-based CLI for Agnitra."""
from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Optional, Sequence, TYPE_CHECKING

import click

from .sdk import optimize as optimize_with_metering, resolve_input_tensor

# Backwards compatibility for older tests/importers expecting ``optimize_model``
optimize_model = optimize_with_metering

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
@click.option(
    "target_label",
    "--target",
    default=None,
    help="Target hardware label (e.g. A100, H100, CPU).",
)
@click.option(
    "offline_mode",
    "--offline",
    is_flag=True,
    help="Run without network calls (enterprise license with offline entitlement required).",
)
@click.option(
    "require_license",
    "--require-license",
    is_flag=True,
    help="Fail if a valid enterprise license is not present.",
)
@click.option(
    "license_seat",
    "--license-seat",
    default=None,
    help="Override seat identifier used for license enforcement.",
)
@click.option(
    "license_org",
    "--license-org",
    default=None,
    help="Override organisation identifier for per-GPU licensing.",
)
def optimize_command(
    model_path: Path,
    input_shape: Optional[Sequence[int]],
    output_path: Optional[Path],
    device_name: Optional[str],
    disable_rl: bool,
    target_label: Optional[str],
    offline_mode: bool,
    require_license: bool,
    license_seat: Optional[str],
    license_org: Optional[str],
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

    metadata = {
        "source": "cli.optimize",
        "device": device_name or "auto",
    }
    if target_label:
        metadata["target"] = target_label
    if hasattr(sample, "shape"):
        metadata["input_shape"] = tuple(int(dim) for dim in sample.shape)  # type: ignore[misc]

    project_id = os.getenv("AGNITRA_PROJECT_ID", "default")
    license_org_id = license_org or os.getenv("AGNITRA_LICENSE_ORG") or project_id

    try:
        result = optimize_with_metering(
            model,
            input_tensor=sample,
            input_shape=None,
            device=device,
            enable_rl=not disable_rl,
            project_id=project_id,
            model_name=model_path.stem,
            metadata=metadata,
            offline=offline_mode,
            require_license=require_license,
            license_seat=license_seat,
            license_org_id=license_org_id,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    optimized = result.optimized_model

    output_path = _resolve_output_path(model_path, output_path)

    try:
        _save_model(optimized, output_path, sample, torch)
    except Exception as exc:
        raise click.ClickException(f"Failed to save optimized model: {exc}") from exc

    click.echo(f"Optimized model written to {output_path}")

    usage_event = getattr(result, "usage_event", None)
    if usage_event is not None:
        click.echo(
            (
                f"Performance uplift: {usage_event.performance_uplift_pct:.1f}% | "
                f"GPU hours saved: {usage_event.gpu_hours_saved:.6f} | "
                f"Billable: {usage_event.total_billable:.4f} {usage_event.currency}"
            )
        )

    license_notes = result.notes.get("license") if isinstance(result.notes, dict) else None
    if isinstance(license_notes, dict):
        seat_id = license_notes.get("seat_id")
        if seat_id:
            click.echo(f"License seat checked out: {seat_id}")
        gpu_usage = license_notes.get("gpu_usage")
        if isinstance(gpu_usage, dict):
            click.echo(
                (
                    f"Tracked GPUs: {gpu_usage.get('active_gpu_count', 0)} "
                    f"| last run timestamp: {gpu_usage.get('timestamp')}"
                )
            )


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
