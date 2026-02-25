"""Optimization pass plugin registry — ClawHub-style skill discovery for Agnitra.

Developers publish custom optimization passes as Python packages and register
them via the ``agnitra.passes`` entry point group. The registry discovers them
at runtime, exactly like OpenClaw discovers skills from ``~/.openclaw/workspace/skills/``.

Publishing a custom pass
------------------------
In your package's ``pyproject.toml``::

    [project.entry-points."agnitra.passes"]
    my_pass = "my_package.passes:MyOptimizationPass"

Your pass class must implement :class:`OptimizationPass`::

    from agnitra.core.optimizer.pass_registry import OptimizationPass

    class MyOptimizationPass(OptimizationPass):
        name = "my_pass"
        description = "My custom optimization pass"

        def apply(self, model, input_tensor, **kwargs):
            # ... transform model ...
            return model

Using the registry
------------------
::

    from agnitra.core.optimizer.pass_registry import PassRegistry

    registry = PassRegistry()
    print(registry.discover())       # ["quantize", "fuse_ops", "my_pass", ...]
    pass_cls = registry.load("my_pass")
    result = pass_cls().apply(model, sample)

Passes can also be loaded from ``~/.agnitra/passes/`` as Python files::

    ~/.agnitra/passes/my_pass.py   # must define class MyPass(OptimizationPass)
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

LOGGER = logging.getLogger(__name__)

_ENTRY_POINT_GROUP = "agnitra.passes"
_USER_PASSES_DIR = Path.home() / ".agnitra" / "passes"


class OptimizationPass(ABC):
    """Base class for all optimization passes.

    Subclass this and implement :meth:`apply` to create a custom pass.
    Register it via the ``agnitra.passes`` entry point group so the
    :class:`PassRegistry` can discover it automatically.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def apply(self, model: Any, input_tensor: Any, **kwargs: Any) -> Any:
        """Apply the optimization pass to *model*.

        Parameters
        ----------
        model:
            The PyTorch module to transform.
        input_tensor:
            A representative input tensor used for tracing/profiling.
        **kwargs:
            Additional pass-specific keyword arguments.

        Returns
        -------
        Any
            The transformed (or original) model.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class PassRegistry:
    """Discover and load optimization pass plugins.

    Discovery order:
    1. Built-in passes (shipped with Agnitra).
    2. Entry point passes registered under ``agnitra.passes``.
    3. Local file passes found in ``~/.agnitra/passes/``.

    Parameters
    ----------
    user_passes_dir:
        Override the directory scanned for local file passes.
    """

    _BUILTIN_PASSES: Dict[str, Type[OptimizationPass]] = {}

    def __init__(self, *, user_passes_dir: Optional[Path] = None) -> None:
        self._user_dir = user_passes_dir or _USER_PASSES_DIR
        self._cache: Optional[Dict[str, Type[OptimizationPass]]] = None

    @classmethod
    def register_builtin(cls, pass_cls: Type[OptimizationPass]) -> Type[OptimizationPass]:
        """Decorator to register a built-in pass class."""
        cls._BUILTIN_PASSES[pass_cls.name] = pass_cls
        return pass_cls

    def discover(self) -> List[str]:
        """Return names of all available passes."""
        return sorted(self._all_passes().keys())

    def load(self, name: str) -> Type[OptimizationPass]:
        """Return the pass class for *name*.

        Raises
        ------
        KeyError
            When no pass with *name* is found.
        """
        passes = self._all_passes()
        if name not in passes:
            available = ", ".join(sorted(passes.keys())) or "none"
            raise KeyError(
                f"Optimization pass {name!r} not found. Available: {available}"
            )
        return passes[name]

    def apply(self, name: str, model: Any, input_tensor: Any, **kwargs: Any) -> Any:
        """Load pass *name* and immediately apply it to *model*."""
        pass_cls = self.load(name)
        return pass_cls().apply(model, input_tensor, **kwargs)

    def info(self) -> List[Dict[str, str]]:
        """Return a list of dicts with name/description for each pass."""
        return [
            {"name": name, "description": cls.description or ""}
            for name, cls in sorted(self._all_passes().items())
        ]

    def _all_passes(self) -> Dict[str, Type[OptimizationPass]]:
        if self._cache is not None:
            return self._cache
        passes: Dict[str, Type[OptimizationPass]] = {}
        passes.update(self._BUILTIN_PASSES)
        passes.update(self._load_entry_point_passes())
        passes.update(self._load_user_file_passes())
        self._cache = passes
        return passes

    def _load_entry_point_passes(self) -> Dict[str, Type[OptimizationPass]]:
        result: Dict[str, Type[OptimizationPass]] = {}
        try:
            eps = importlib.metadata.entry_points(group=_ENTRY_POINT_GROUP)
        except Exception as exc:
            LOGGER.debug("Could not read entry points for %s: %s", _ENTRY_POINT_GROUP, exc)
            return result

        for ep in eps:
            try:
                cls = ep.load()
                if not (isinstance(cls, type) and issubclass(cls, OptimizationPass)):
                    LOGGER.warning(
                        "Entry point %r does not implement OptimizationPass; skipping.", ep.name
                    )
                    continue
                name = getattr(cls, "name", None) or ep.name
                result[name] = cls
                LOGGER.debug("Loaded pass %r from entry point %r.", name, ep.value)
            except Exception as exc:
                LOGGER.warning("Failed to load pass from entry point %r: %s", ep.name, exc)

        return result

    def _load_user_file_passes(self) -> Dict[str, Type[OptimizationPass]]:
        result: Dict[str, Type[OptimizationPass]] = {}
        if not self._user_dir.exists():
            return result

        for py_file in sorted(self._user_dir.glob("*.py")):
            module_name = f"_agnitra_user_pass_{py_file.stem}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)  # type: ignore[attr-defined]
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)  # type: ignore[attr-defined]
                sys.modules[module_name] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, OptimizationPass)
                        and attr is not OptimizationPass
                        and getattr(attr, "name", "")
                    ):
                        result[attr.name] = attr
                        LOGGER.debug("Loaded user pass %r from %s.", attr.name, py_file)
            except Exception as exc:
                LOGGER.warning("Failed to load user pass from %s: %s", py_file, exc)

        return result


# --- Built-in pass: identity (no-op, useful for testing) ---

@PassRegistry.register_builtin
class IdentityPass(OptimizationPass):
    """No-op pass that returns the model unchanged. Useful for testing."""

    name = "identity"
    description = "No-op pass — returns the model unchanged."

    def apply(self, model: Any, input_tensor: Any, **kwargs: Any) -> Any:
        return model


__all__ = ["OptimizationPass", "PassRegistry"]
