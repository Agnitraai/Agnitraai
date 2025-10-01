"""Runtime patch injector utilities for Agnitra.

This module provides a high-level wrapper around ``torch.fx`` graph rewriting
and module hooks to inject custom kernel wrappers at runtime. The primary entry
point, :class:`RuntimePatcher`, orchestrates three responsibilities:

* locate FX nodes that correspond to expensive operators and replace them with
  callable wrappers that execute an optimized kernel with automatic fallback to
  the baseline operation when failures occur;
* attach lightweight ``register_forward_hook`` callbacks when FX rewriting is
  undesirable or unavailable (e.g., tracing a dynamic control-flow module);
* capture structured patch logs so downstream tooling and demos can surface a
  friendly summary of which optimizations were applied.

All APIs are best-effort: if Torch is missing or a specific patch fails, the
original model remains usable. Warnings are emitted to aid debugging, but
exceptions only propagate for irrecoverable configuration mistakes (such as
requesting FX rewriting without ``torch.fx``).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:  # Optional dependency; tests cover fallback paths when torch isn't present.
    import torch
    from torch import nn
    from torch.fx import GraphModule, symbolic_trace
    from torch.fx.graph import Node
except Exception:  # pragma: no cover - exercised when torch unavailable
    torch = None  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment]
    GraphModule = Any  # type: ignore[assignment]
    symbolic_trace = None  # type: ignore[assignment]
    Node = Any  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)

ValidatorFn = Callable[[Any, Tuple[Any, ...], Mapping[str, Any]], bool]
ForwardHookFn = Callable[[Any, Tuple[Any, ...], Any], Any]


@dataclass(frozen=True)
class FXNodePatch:
    """Describe a Torch FX node replacement strategy."""

    name: str
    target: str
    kernel: Callable[..., Any]
    match_kind: Optional[str] = None
    predicate: Optional[Callable[[Node], bool]] = None
    validator: Optional[ValidatorFn] = None
    fallback: Optional[Callable[..., Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardHookPatch:
    """Describe a hook-based injection on a specific ``nn.Module`` path."""

    name: str
    module_path: str
    kernel: ForwardHookFn
    fallback: Optional[ForwardHookFn] = None
    validator: Optional[ValidatorFn] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PatchLog:
    """Audit entry recorded for each attempted patch."""

    name: str
    strategy: str
    status: str
    detail: str = ""
    matched: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RuntimePatchReport:
    """Aggregate outcome returned by :class:`RuntimePatcher.patch`."""

    module: Any
    logs: Tuple[PatchLog, ...]
    graph_module: Optional[Any] = None
    handles: Tuple[Any, ...] = ()

    @property
    def applied(self) -> Tuple[PatchLog, ...]:
        """Return the subset of logs where a patch was successfully applied."""

        return tuple(entry for entry in self.logs if entry.status == "applied")

    @property
    def skipped(self) -> Tuple[PatchLog, ...]:
        """Return the subset of logs describing skipped patches."""

        return tuple(entry for entry in self.logs if entry.status == "skipped")

    @property
    def errors(self) -> Tuple[PatchLog, ...]:
        """Return the subset of logs describing errored patches."""

        return tuple(entry for entry in self.logs if entry.status == "error")


class RuntimePatcher:
    """Runtime kernel injector with FX rewriting and hook fallbacks."""

    def __init__(self, *, default_copy: bool = False) -> None:
        self._default_copy = bool(default_copy)
        self._fx_wrapper_index = 0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def patch(
        self,
        module: Any,
        *,
        fx_patches: Optional[Sequence[FXNodePatch]] = None,
        hook_patches: Optional[Sequence[ForwardHookPatch]] = None,
        copy_module: Optional[bool] = None,
    ) -> RuntimePatchReport:
        """Apply the requested patches to ``module`` and return a report.

        Parameters
        ----------
        module:
            The ``nn.Module`` (or ``GraphModule``) to patch. When ``copy_module``
            is true a deep copy is made before any transformation so the
            original graph remains untouched.
        fx_patches:
            Optional sequence of :class:`FXNodePatch` entries describing FX node
            replacements.
        hook_patches:
            Optional sequence of :class:`ForwardHookPatch` entries describing
            hook-based injections.
        copy_module:
            Overrides the default set during object construction. When ``None``
            the constructor's ``default_copy`` value is honoured.
        """

        fx_patches = tuple(fx_patches or ())
        hook_patches = tuple(hook_patches or ())

        if torch is None:
            raise RuntimeError("Runtime patching requires PyTorch to be installed.")

        work_module = module
        if copy_module is None:
            copy_module = self._default_copy
        if copy_module:
            work_module = copy.deepcopy(module)

        logs: List[PatchLog] = []
        fx_module: Optional[Any] = None

        if fx_patches:
            if symbolic_trace is None:
                raise RuntimeError("torch.fx is unavailable; cannot apply FX patches.")
            work_module, fx_logs = self._apply_fx_patches(work_module, fx_patches)
            logs.extend(fx_logs)
            if isinstance(work_module, GraphModule):
                fx_module = work_module

        handles: List[Any] = []
        if hook_patches:
            hook_handles, hook_logs = self._apply_hook_patches(work_module, hook_patches)
            handles.extend(hook_handles)
            logs.extend(hook_logs)

        report = RuntimePatchReport(
            module=work_module,
            graph_module=fx_module,
            logs=tuple(logs),
            handles=tuple(handles),
        )
        setattr(work_module, "_agnitra_runtime_patch_log", report.logs)
        if handles:
            setattr(work_module, "_agnitra_runtime_patch_handles", report.handles)
        return report

    def describe_kernel(self, kernel: Any) -> str:
        """Return a human-friendly descriptor for generated kernels.

        This helper preserves backwards compatibility with older demos that
        printed a stub string for generated kernels. Callers may ignore it when
        using the richer :meth:`patch` API.
        """

        descriptor: str
        try:
            from agnitra.core.kernel import KernelGenerationResult  # Local import to avoid cycle
        except Exception:  # pragma: no cover - defensive; import should succeed in normal runs
            KernelGenerationResult = None  # type: ignore[assignment]

        if KernelGenerationResult is not None and isinstance(kernel, KernelGenerationResult):
            descriptor = Path(kernel.module_path).name
        else:
            descriptor = str(kernel)
        return f"Patched<{descriptor}>"

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_fx_patches(
        self,
        module: Any,
        patches: Sequence[FXNodePatch],
    ) -> Tuple[Any, List[PatchLog]]:
        """Trace ``module`` with FX, apply patches, and return (module, logs)."""

        fx_logs: List[PatchLog] = []
        try:
            graph_module = module if isinstance(module, GraphModule) else symbolic_trace(module)
        except Exception as exc:
            raise RuntimeError(f"FX tracing failed: {exc}") from exc

        wrappers: List[Callable[..., Any]] = []
        for patch in patches:
            try:
                matched_nodes = self._locate_nodes(graph_module, patch)
            except Exception as exc:  # pragma: no cover - defensive logging
                LOGGER.exception("Failed to locate nodes for patch %s", patch.name)
                fx_logs.append(
                    PatchLog(
                        name=patch.name,
                        strategy="fx",
                        status="error",
                        detail=str(exc),
                        metadata=patch.metadata,
                    )
                )
                continue

            if not matched_nodes:
                fx_logs.append(
                    PatchLog(
                        name=patch.name,
                        strategy="fx",
                        status="skipped",
                        detail=f"No FX nodes matched target '{patch.target}'",
                        metadata=patch.metadata,
                    )
                )
                continue

            applied_nodes: List[str] = []
            for node in matched_nodes:
                try:
                    fallback = patch.fallback or self._make_fx_fallback(graph_module, node)
                    wrapper = self._build_fx_wrapper(
                        patch.name,
                        patch.kernel,
                        fallback,
                        validator=patch.validator,
                        metadata=patch.metadata,
                    )
                    new_node = self._replace_node_with_wrapper(graph_module, node, wrapper)
                    applied_nodes.append(new_node.name)
                    wrappers.append(wrapper)
                except Exception as exc:
                    LOGGER.warning(
                        "Patch '%s' failed on node '%s'; using fallback. Error: %s",
                        patch.name,
                        node.name,
                        exc,
                    )
                    fx_logs.append(
                        PatchLog(
                            name=patch.name,
                            strategy="fx",
                            status="error",
                            detail=f"Node '{node.name}' replacement failed: {exc}",
                            metadata=patch.metadata,
                        )
                    )
                    break
            else:
                fx_logs.append(
                    PatchLog(
                        name=patch.name,
                        strategy="fx",
                        status="applied",
                        matched=tuple(applied_nodes),
                        metadata=patch.metadata,
                    )
                )

        if wrappers:
            existing = list(getattr(graph_module, "_agnitra_fx_wrappers", []))
            existing.extend(wrappers)
            setattr(graph_module, "_agnitra_fx_wrappers", existing)

        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module, fx_logs

    def _locate_nodes(self, graph_module: GraphModule, patch: FXNodePatch) -> List[Node]:
        """Return FX nodes that match ``patch`` criteria."""

        matches: List[Node] = []
        for node in graph_module.graph.nodes:
            if node.op in {"placeholder", "output"}:
                continue
            if patch.match_kind and node.op != patch.match_kind:
                continue
            if not self._matches_target(node, patch.target):
                continue
            if patch.predicate and not patch.predicate(node):
                continue
            matches.append(node)
        return matches

    def _matches_target(self, node: Node, target: str) -> bool:
        """Return ``True`` when ``target`` matches the FX node identifier."""

        aliases = {node.name, str(node.target)}
        target_obj = node.target
        if isinstance(target_obj, str):
            aliases.add(target_obj)
        module_name = getattr(target_obj, "__module__", None)
        qualname = getattr(target_obj, "__qualname__", None)
        func_name = getattr(target_obj, "__name__", None)
        for candidate in (qualname, func_name):
            if candidate:
                aliases.add(candidate)
                if module_name:
                    aliases.add(f"{module_name}.{candidate}")
        if module_name:
            aliases.add(module_name)

        normalized: set[str] = set()
        for alias in aliases:
            if not alias:
                continue
            normalized.add(alias)
            sanitized = alias.replace("::", ".")
            normalized.add(sanitized)
            normalized.add(alias.lstrip("_"))
            normalized.add(sanitized.lstrip("_"))
            if "." in sanitized:
                normalized.add(sanitized.split(".")[-1])
        normalized = {item for item in normalized if item}
        if target in normalized:
            return True
        if target.startswith("aten::"):
            suffix = target.split("::")[-1]
            return any(item.endswith(suffix) for item in normalized)
        return False

    def _make_fx_fallback(self, module: GraphModule, node: Node) -> Callable[..., Any]:
        """Construct a callable that replays the original FX node target."""

        if node.op == "call_function":
            target = node.target

            def fallback(*args: Any, **kwargs: Any) -> Any:
                return target(*args, **kwargs)

        elif node.op == "call_method":
            method_name = str(node.target)

            def fallback(*args: Any, **kwargs: Any) -> Any:
                owner, *rest = args
                method = getattr(owner, method_name)
                return method(*rest, **kwargs)

        elif node.op == "call_module":
            submodule = module.get_submodule(str(node.target))

            def fallback(*args: Any, **kwargs: Any) -> Any:
                return submodule(*args, **kwargs)

        else:  # pragma: no cover - defensive branch for future FX ops
            raise NotImplementedError(f"Unsupported FX node op: {node.op}")

        return fallback

    def _build_fx_wrapper(
        self,
        name: str,
        kernel: Callable[..., Any],
        fallback: Callable[..., Any],
        *,
        validator: Optional[ValidatorFn],
        metadata: Mapping[str, Any],
    ) -> Callable[..., Any]:
        """Return a callable that wraps ``kernel`` with fallback + validation."""

        index = self._fx_wrapper_index
        self._fx_wrapper_index += 1
        wrapper_name = f"agnitra_fx_patch_{index}_{name.replace(' ', '_')}"

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                result = kernel(*args, **kwargs)
            except Exception as exc:
                LOGGER.warning(
                    "Optimized kernel '%s' raised %s; using fallback.",
                    name,
                    exc,
                )
                return fallback(*args, **kwargs)
            if validator is not None:
                try:
                    ok = validator(result, args, kwargs)
                except Exception as exc:  # pragma: no cover - validator error path
                    LOGGER.warning(
                        "Validator for '%s' failed (%s); reverting to fallback.",
                        name,
                        exc,
                    )
                    return fallback(*args, **kwargs)
                if not ok:
                    LOGGER.warning(
                        "Validator for '%s' returned False; reverting to fallback.",
                        name,
                    )
                    return fallback(*args, **kwargs)
            return result

        wrapper.__name__ = wrapper_name
        setattr(wrapper, "_agnitra_patch", {"name": name, "strategy": "fx", "metadata": dict(metadata)})
        return wrapper

    def _replace_node_with_wrapper(
        self,
        module: GraphModule,
        node: Node,
        wrapper: Callable[..., Any],
    ) -> Node:
        """Replace ``node`` with a call to ``wrapper`` and return the new node."""

        graph = module.graph
        with graph.inserting_after(node):
            new_node = graph.call_function(wrapper, args=tuple(node.args), kwargs=dict(node.kwargs))
            new_node.meta.update(getattr(node, "meta", {}))
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        return new_node

    def _apply_hook_patches(
        self,
        module: Any,
        patches: Sequence[ForwardHookPatch],
    ) -> Tuple[List[Any], List[PatchLog]]:
        """Attach forward hooks for ``patches`` and return (handles, logs)."""

        handles: List[Any] = []
        logs: List[PatchLog] = []
        for patch in patches:
            try:
                target_module = module.get_submodule(patch.module_path)
            except AttributeError as exc:
                logs.append(
                    PatchLog(
                        name=patch.name,
                        strategy="hook",
                        status="skipped",
                        detail=f"Module path '{patch.module_path}' not found: {exc}",
                        metadata=patch.metadata,
                    )
                )
                continue

            fallback = patch.fallback or (lambda _m, _inputs, output: output)
            hook = self._build_forward_hook(
                patch.name,
                patch.kernel,
                fallback,
                validator=patch.validator,
                metadata=patch.metadata,
            )
            handle = target_module.register_forward_hook(hook)
            handles.append(handle)
            logs.append(
                PatchLog(
                    name=patch.name,
                    strategy="hook",
                    status="applied",
                    matched=(patch.module_path,),
                    metadata=patch.metadata,
                )
            )
        return handles, logs

    def _build_forward_hook(
        self,
        name: str,
        kernel: ForwardHookFn,
        fallback: ForwardHookFn,
        *,
        validator: Optional[ValidatorFn],
        metadata: Mapping[str, Any],
    ) -> ForwardHookFn:
        """Wrap a forward hook kernel with fallback and validation."""

        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> Any:  # type: ignore[override]
            try:
                result = kernel(module, inputs, output)
            except Exception as exc:
                LOGGER.warning(
                    "Forward hook '%s' raised %s; using fallback.",
                    name,
                    exc,
                )
                return fallback(module, inputs, output)
            if validator is not None:
                try:
                    ok = validator(result, inputs, {})
                except Exception as exc:  # pragma: no cover - validator error path
                    LOGGER.warning(
                        "Forward hook validator for '%s' failed (%s); using fallback.",
                        name,
                        exc,
                    )
                    return fallback(module, inputs, output)
                if not ok:
                    LOGGER.warning(
                        "Forward hook validator for '%s' returned False; using fallback.",
                        name,
                    )
                    return fallback(module, inputs, output)
            return result

        setattr(hook, "_agnitra_patch", {"name": name, "strategy": "hook", "metadata": dict(metadata)})
        return hook


__all__ = [
    "FXNodePatch",
    "ForwardHookPatch",
    "PatchLog",
    "RuntimePatchReport",
    "RuntimePatcher",
]
