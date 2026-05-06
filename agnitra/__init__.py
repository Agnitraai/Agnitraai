"""Agnitra SDK package.

The package import is intentionally cheap: ``import agnitra`` reads
``__version__`` and exits without touching torch, transformers, or any
of the optimizer machinery. Heavy attributes (``optimize``,
``optimize_model``, the ``sdk`` submodule) are loaded lazily on first
attribute access via PEP 562 ``__getattr__``.

This matters because users often `pip install agnitra` and run
`agnitra --help` or `import agnitra` before they've finished setting
up CUDA / torch. A heavy import here would surface as a confusing
ImportError; lazy access surfaces the dependency requirement at the
moment the user actually invokes optimize() — where the error message
can be precise.
"""
from importlib import metadata

try:
    __version__ = metadata.version("agnitra")
except metadata.PackageNotFoundError:  # pragma: no cover - local source checkout
    __version__ = "0.0.0"


_LAZY_ATTRS = frozenset({"optimize", "optimize_model", "sdk"})

__all__ = ["optimize", "optimize_model", "sdk", "__version__"]


def __getattr__(name: str):  # noqa: D401 - PEP 562 module __getattr__
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module 'agnitra' has no attribute {name!r}")
    import importlib
    # Use importlib directly to avoid triggering this __getattr__ recursively
    # — `from . import sdk` would call getattr(agnitra, "sdk") which loops.
    _sdk_module = importlib.import_module("agnitra.sdk")
    if name == "sdk":
        value = _sdk_module
    else:
        value = getattr(_sdk_module, name)
    # Cache for subsequent lookups so we only pay the import cost once.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _LAZY_ATTRS | {"__version__"})
