"""Agnitra SDK package."""

from importlib import metadata

from . import sdk  # Re-export submodule for advanced usage.
from .sdk import optimize, optimize_model

try:
    __version__ = metadata.version("agnitra")
except metadata.PackageNotFoundError:  # pragma: no cover - local source checkout
    __version__ = "0.0.0"

__all__ = ["optimize", "optimize_model", "sdk", "__version__"]
