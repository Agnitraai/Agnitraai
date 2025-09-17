"""FX-based IR graph extractor with telemetry annotations.

This module traces PyTorch ``nn.Module`` instances using ``torch.fx`` and
produces a JSON-serialisable intermediate representation enriched with
telemetry signals collected during profiling. It validates basic shape
information to ensure end users receive a trustworthy IR snapshot that can be
consumed by downstream optimisation passes (LLM heuristics, RL agents, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import torch
    from torch.fx import GraphModule, Node, symbolic_trace
    from torch.fx.passes.shape_prop import ShapeProp
except Exception:  # pragma: no cover - exercised when torch/fx absent
    torch = None
    GraphModule = Node = ShapeProp = symbolic_trace = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class IRNode:
    """Structured node representation used for serialisation."""

    name: str
    op: str
    target: str
    kind: str
    shape: Optional[Any]
    dtype: Optional[str]
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    cuda_time_ms: Optional[float]
    cpu_time_ms: Optional[float]
    memory_bytes: Optional[int]
    telemetry_sources: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "op": self.op,
            "target": self.target,
            "kind": self.kind,
            "shape": self.shape,
            "dtype": self.dtype,
            "input_shapes": self.input_shapes,
            "output_shapes": self.output_shapes,
            "cuda_time_ms": self.cuda_time_ms,
            "cpu_time_ms": self.cpu_time_ms,
            "memory_bytes": self.memory_bytes,
            "telemetry_sources": self.telemetry_sources,
        }


def _requires_torch_fx() -> None:
    if torch is None or symbolic_trace is None or GraphModule is None:
        raise RuntimeError("torch.fx is required for graph extraction but is unavailable")


def _ensure_tuple_inputs(example_inputs: Any | Sequence[Any] | None) -> Tuple[Any, ...]:
    if example_inputs is None:
        return tuple()
    if isinstance(example_inputs, tuple):
        return example_inputs
    if isinstance(example_inputs, list):
        return tuple(example_inputs)
    return (example_inputs,)


def _normalise_name(name: Any) -> str:
    s = str(name or "").strip()
    if not s:
        return ""
    lowered = s.lower()
    for prefix in ("aten::", "torch.", "builtin::", "operator."):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix) :]
            break
    lowered = lowered.split("(", 1)[0].strip()
    return lowered


def _dtype_str(dtype: Any) -> Optional[str]:
    if dtype is None:
        return None
    try:
        text = str(dtype)
        if text.startswith("torch."):
            return text.split(".", 1)[1]
        return text
    except Exception:
        return None


def _meta_produces_tensor(meta: Any) -> bool:
    if meta is None:
        return False
    if isinstance(meta, (list, tuple)):
        return any(_meta_produces_tensor(m) for m in meta)
    return hasattr(meta, "shape") and getattr(meta, "shape") is not None


def _extract_shapes(meta: Any) -> List[List[int]]:
    shapes: List[List[int]] = []

    def _collect(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, (list, tuple)):
            for item in obj:
                _collect(item)
            return
        shape = getattr(obj, "shape", None)
        if isinstance(shape, (list, tuple)):
            try:
                shapes.append([int(x) for x in shape])
            except Exception:
                pass
        elif torch is not None and isinstance(obj, torch.Tensor):  # pragma: no cover - guard
            shapes.append([int(x) for x in obj.shape])

    _collect(meta)
    return shapes


def _extract_dtype(meta: Any) -> Optional[str]:
    if meta is None:
        return None
    if isinstance(meta, (list, tuple)) and meta:
        for item in meta:
            dtype = _extract_dtype(item)
            if dtype is not None:
                return dtype
        return None
    dtype = getattr(meta, "dtype", None)
    if dtype is None and torch is not None and isinstance(meta, torch.Tensor):  # pragma: no cover - guard
        dtype = meta.dtype
    return _dtype_str(dtype)


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _microseconds_to_ms(value: Any) -> Optional[float]:
    val = _to_float(value)
    return (val / 1000.0) if val is not None else None


class _TelemetryIndex:
    """Index telemetry payloads for fast lookup during annotation."""

    def __init__(self, telemetry: Optional[Dict[str, Any]]):
        self._layer_by_name: Dict[str, Dict[str, Any]] = {}
        self._event_by_key: Dict[str, Dict[str, Any]] = {}
        if not isinstance(telemetry, dict):
            return
        for layer in telemetry.get("model", {}).get("layers", []):
            name = layer.get("name")
            if isinstance(name, str) and name not in self._layer_by_name:
                self._layer_by_name[name] = layer
        for event in telemetry.get("events", []):
            key = _normalise_name(event.get("name"))
            if key and key not in self._event_by_key:
                self._event_by_key[key] = event

    @property
    def layer_names(self) -> Iterable[str]:
        return self._layer_by_name.keys()

    def layer(self, name: str) -> Optional[Dict[str, Any]]:
        return self._layer_by_name.get(name)

    def event(self, op_name: str) -> Optional[Dict[str, Any]]:
        key = _normalise_name(op_name)
        if key in self._event_by_key:
            return self._event_by_key[key]
        return None


class GraphIRExtractor:
    """Extract FX IR graphs enriched with telemetry context."""

    def __init__(self, telemetry: Optional[Dict[str, Any]] = None, validate: bool = True) -> None:
        _requires_torch_fx()
        self._telemetry = _TelemetryIndex(telemetry)
        self._validate = validate

    def extract(self, model: Any, example_inputs: Any | Sequence[Any] | None = None) -> List[Dict[str, Any]]:
        traced = symbolic_trace(model)
        self._propagate_shapes(traced, example_inputs)
        nodes: List[IRNode] = []
        module_nodes_seen: set[str] = set()
        missing_shapes: List[str] = []

        for node in traced.graph.nodes:
            if node.op == "output":
                continue

            ir_node, produced_tensor = self._serialize_node(node, traced)
            if node.op == "call_module" and node.target:
                module_nodes_seen.add(str(node.target))

            if self._validate and produced_tensor and ir_node.shape is None:
                missing_shapes.append(ir_node.name)

            nodes.append(ir_node)

        if self._validate:
            self._validate_coverage(traced, module_nodes_seen)
            if missing_shapes:
                raise ValueError(
                    "Missing shape information for tensor-producing nodes: "
                    + ", ".join(sorted(set(missing_shapes)))
                )

        return [node.to_dict() for node in nodes]

    def _propagate_shapes(self, traced: GraphModule, example_inputs: Any | Sequence[Any] | None) -> None:
        if ShapeProp is None:
            return
        args = _ensure_tuple_inputs(example_inputs)
        if not args:
            return
        try:
            with torch.no_grad():  # type: ignore[union-attr]
                ShapeProp(traced).propagate(*args)
        except Exception as exc:  # pragma: no cover - diagnostics only
            LOGGER.debug("Shape propagation failed: %s", exc)

    def _serialize_node(self, node: Node, traced: GraphModule) -> Tuple[IRNode, bool]:
        meta = node.meta.get("tensor_meta") if hasattr(node, "meta") else None
        if meta is None:
            meta = node.meta.get("val") if hasattr(node, "meta") else None
        shapes = _extract_shapes(meta)
        dtype = _extract_dtype(meta)
        produced_tensor = _meta_produces_tensor(meta)
        if not produced_tensor and node.op in {"call_module", "call_function", "call_method"}:
            produced_tensor = True

        module = traced.get_submodule(node.target) if node.op == "call_module" and node.target else None
        module_name = str(node.target) if node.target else ""
        module_type = module.__class__.__name__ if module is not None else ""

        layer_info = self._telemetry.layer(module_name)
        event_info = None
        telemetry_sources: List[str] = []
        cuda_time_ms: Optional[float] = None
        cpu_time_ms: Optional[float] = None
        memory_bytes: Optional[int] = None

        if layer_info:
            telemetry_sources.append("layer")
            if not shapes:
                shapes = [list(s) for s in layer_info.get("output_shapes", []) if isinstance(s, list)]
            dtype = dtype or _dtype_str(layer_info.get("output_dtype"))
            cuda_time_ms = _to_float(layer_info.get("forward_time_ms"))
            memory_bytes = memory_bytes or layer_info.get("cuda_mem_alloc_delta_bytes")
            if cuda_time_ms is not None:
                cuda_time_ms = max(cuda_time_ms, 0.0)

        if node.op == "call_module":
            op_display = module_type or module_name or node.name
            search_keys = [module_name, module_type]
        elif node.op == "call_function":
            op_display = getattr(node.target, "__name__", str(node.target))
            search_keys = [getattr(node.target, "__name__", None), node.name, op_display]
        elif node.op == "call_method":
            op_display = str(node.target)
            search_keys = [node.target, node.name]
        else:
            op_display = str(node.target)
            search_keys = [node.target, node.name]

        if node.op != "call_module" or layer_info is None:
            for key in search_keys:
                if key is None:
                    continue
                event_info = self._telemetry.event(key)
                if event_info:
                    break

        if event_info:
            telemetry_sources.append("event")
            cuda_candidate = _microseconds_to_ms(event_info.get("cuda_time_avg") or event_info.get("cuda_time_total"))
            cpu_candidate = _microseconds_to_ms(event_info.get("cpu_time_avg") or event_info.get("cpu_time_total"))
            mem_candidate = event_info.get("self_cuda_memory_usage")
            if cuda_candidate is not None:
                cuda_time_ms = cuda_candidate if cuda_time_ms is None else max(cuda_time_ms, cuda_candidate)
            if cpu_candidate is not None:
                cpu_time_ms = cpu_candidate
            if memory_bytes is None and isinstance(mem_candidate, (int, float)):
                memory_bytes = int(mem_candidate)

        shape_value: Optional[Any]
        if not shapes:
            shape_value = None
        elif len(shapes) == 1:
            shape_value = shapes[0]
        else:
            shape_value = shapes

        input_shapes = layer_info.get("input_shapes", []) if layer_info else []
        input_shapes = [list(s) for s in input_shapes if isinstance(s, list)]

        ir_node = IRNode(
            name=str(node.name),
            op=str(_normalise_name(op_display) or op_display),
            target=str(node.target),
            kind=str(node.op),
            shape=shape_value,
            dtype=dtype,
            input_shapes=input_shapes,
            output_shapes=[shape for shape in shapes],
            cuda_time_ms=cuda_time_ms,
            cpu_time_ms=cpu_time_ms,
            memory_bytes=int(memory_bytes) if isinstance(memory_bytes, (int, float)) else None,
            telemetry_sources=telemetry_sources,
        )
        return ir_node, produced_tensor

    def _validate_coverage(self, traced: GraphModule, module_nodes_seen: set[str]) -> None:
        expected_modules = {name for name, _ in traced.named_modules() if name}
        missing = expected_modules - module_nodes_seen
        if missing:
            raise ValueError(
                "Telemetry/graph coverage mismatch for modules: " + ", ".join(sorted(missing))
            )
        telemetry_layers = {name for name in self._telemetry.layer_names if name}
        missing_layers = telemetry_layers - module_nodes_seen
        if missing_layers:
            raise ValueError(
                "Telemetry data provided for modules absent in FX graph: "
                + ", ".join(sorted(missing_layers))
            )


def extract_graph_ir(
    model: Any,
    example_inputs: Any | Sequence[Any] | None = None,
    telemetry: Optional[Dict[str, Any]] = None,
    validate: bool = True,
) -> List[Dict[str, Any]]:
    """Convenience wrapper returning IR dictionaries for ``model``.

    Parameters
    ----------
    model:
        ``torch.nn.Module`` instance to trace.
    example_inputs:
        Example positional inputs used for ``ShapeProp`` to populate shapes.
    telemetry:
        Optional telemetry payload as produced by ``telemetry_collector``.
    validate:
        When ``True`` (default), ensure tensor-producing nodes have shapes and
        that traced modules are covered by the resulting IR.
    """

    extractor = GraphIRExtractor(telemetry=telemetry, validate=validate)
    return extractor.extract(model, example_inputs=example_inputs)


def extract_fx_ir(
    model: Any,
    telemetry: Optional[Dict[str, Any]] = None,
    example_inputs: Any | Sequence[Any] | None = None,
    validate: bool = True,
) -> List[Dict[str, Any]]:
    """Backward-compatible alias for :func:`extract_graph_ir`."""

    return extract_graph_ir(
        model=model,
        example_inputs=example_inputs,
        telemetry=telemetry,
        validate=validate,
    )


__all__ = ["GraphIRExtractor", "extract_graph_ir", "extract_fx_ir", "IRNode"]
