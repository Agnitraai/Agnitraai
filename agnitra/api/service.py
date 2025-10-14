"""Agentic optimization pipeline helpers."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from agnitra.core.kernel import KernelGenerator


def run_agentic_optimization(
    model_graph: Any,
    telemetry: Any,
    target: str,
    *,
    kernel_generator: Optional[KernelGenerator] = None,
) -> Dict[str, Any]:
    """Compute optimization recommendations for ``model_graph`` on ``target``.

    Parameters
    ----------
    model_graph:
        JSON-like payload describing the profiled model graph. Accepts either a
        list of node dicts or a mapping containing a ``nodes`` sequence.
    telemetry:
        JSON-like payload describing profiler telemetry. Accepts either a list
        of event dicts or a mapping containing an ``events`` sequence.
    target:
        Target accelerator (e.g. ``"A100"``) guiding the tuning heuristics.
    kernel_generator:
        Optional :class:`KernelGenerator` instance used to render Triton code.

    Returns
    -------
    dict
        Structured payload containing keys ``ir_graph``, ``kernel``, and
        ``patch_instructions``.
    """

    target_normalised = (target or "").strip()
    if not target_normalised:
        raise ValueError("target must be provided")

    nodes = _normalize_graph(model_graph)
    if not nodes:
        raise ValueError("model_graph is missing graph nodes")

    events = _normalize_telemetry(telemetry)
    telemetry_summary = _summarize_events(events)

    focus_node = _select_bottleneck_node(nodes, telemetry_summary)

    generator = kernel_generator or KernelGenerator()
    kernel_payload = _generate_kernel_payload(generator, focus_node, target_normalised)

    patch_plan = _build_patch_plan(
        focus_node=focus_node,
        target=target_normalised,
        telemetry_summary=telemetry_summary,
        kernel_payload=kernel_payload,
    )

    optimized_graph = _craft_optimized_graph(nodes, focus_node, patch_plan, target_normalised)

    return {
        "target": target_normalised,
        "telemetry_summary": telemetry_summary,
        "bottleneck": {
            "name": focus_node.get("name"),
            "op": focus_node.get("op"),
            "baseline_latency_ms": patch_plan["metrics"]["baseline_latency_ms"],
            "expected_latency_ms": patch_plan["metrics"]["expected_latency_ms"],
            "expected_speedup_pct": patch_plan["metrics"]["expected_speedup_pct"],
        },
        "ir_graph": {
            "nodes": optimized_graph,
            "metadata": {
                "target": target_normalised,
                "node_count": len(optimized_graph),
                "events_seen": telemetry_summary.get("event_count"),
                "total_latency_ms": telemetry_summary.get("total_latency_ms"),
            },
        },
        "kernel": kernel_payload,
        "patch_instructions": patch_plan["instructions"],
    }


def _normalize_graph(model_graph: Any) -> List[Dict[str, Any]]:
    """Return a list of graph nodes from ``model_graph``."""

    if isinstance(model_graph, Mapping):
        maybe_nodes = model_graph.get("nodes")
        if isinstance(maybe_nodes, Sequence):
            return [
                dict(_ensure_node_name(node, idx))
                for idx, node in enumerate(maybe_nodes)
                if isinstance(node, Mapping)
            ]
        return [
            dict(_ensure_node_name(value, idx))
            for idx, value in enumerate(model_graph.values())
            if isinstance(value, Mapping)
        ]

    if isinstance(model_graph, Sequence):
        return [
            dict(_ensure_node_name(node, idx))
            for idx, node in enumerate(model_graph)
            if isinstance(node, Mapping)
        ]

    return []


def _ensure_node_name(node: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    """Ensure each node has a stable ``name`` attribute."""

    if "name" in node and isinstance(node["name"], str) and node["name"].strip():
        return node
    derived = dict(node)
    derived["name"] = f"node_{index}"
    return derived


def _normalize_telemetry(telemetry: Any) -> List[Dict[str, Any]]:
    """Return profiler events from ``telemetry``."""

    if telemetry is None:
        return []

    if isinstance(telemetry, Mapping):
        maybe_events = telemetry.get("events")
        if isinstance(maybe_events, Sequence):
            return [dict(event) for event in maybe_events if isinstance(event, Mapping)]
        # Accept raw event payloads keyed by name
        return [
            dict(event)
            for event in telemetry.values()
            if isinstance(event, Mapping)
        ]

    if isinstance(telemetry, Sequence):
        return [dict(event) for event in telemetry if isinstance(event, Mapping)]

    return []


def _summarize_events(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate latency stats for telemetry events."""

    total_latency = 0.0
    max_latency = 0.0
    latencies: List[float] = []
    bottleneck_event: Optional[Dict[str, Any]] = None

    for event in events:
        latency = _first_float(
            event,
            ("cuda_time_total", "cuda_time_ms", "latency_ms", "cpu_time_total", "cpu_time_ms"),
        )
        if latency is None:
            continue
        latency = max(0.0, latency)
        total_latency += latency
        latencies.append(latency)
        if latency >= max_latency:
            max_latency = latency
            bottleneck_event = dict(event)

    avg_latency = (total_latency / len(latencies)) if latencies else 0.0

    return {
        "event_count": len(events),
        "total_latency_ms": round(total_latency, 6),
        "average_latency_ms": round(avg_latency, 6),
        "max_latency_ms": round(max_latency, 6),
        "bottleneck_event": bottleneck_event,
    }


def _select_bottleneck_node(
    nodes: Sequence[Mapping[str, Any]],
    telemetry_summary: Mapping[str, Any],
) -> Dict[str, Any]:
    """Pick the graph node with the highest latency contribution."""

    fallback_latency = float(telemetry_summary.get("max_latency_ms") or 0.0)
    selected: Optional[Mapping[str, Any]] = None
    selected_latency = -math.inf

    for node in nodes:
        latency = _node_latency(node)
        if latency is None:
            latency = fallback_latency
        if latency > selected_latency:
            selected = node
            selected_latency = latency

    if selected is None:
        raise ValueError("Unable to select a bottleneck node from model_graph")

    return dict(selected)


def _node_latency(node: Mapping[str, Any]) -> Optional[float]:
    """Return the best-effort latency estimate for ``node``."""

    latency = _first_float(node, ("cuda_time_ms", "latency_ms", "cpu_time_ms"))
    if latency is None:
        return None
    return max(0.0, latency)


def _first_float(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
) -> Optional[float]:
    """Return the first key in ``keys`` coercible to ``float``."""

    for key in keys:
        value = mapping.get(key)
        try:
            if value is None:
                continue
            num = float(value)
            if math.isfinite(num):
                return num
        except (TypeError, ValueError):
            continue
    return None


def _generate_kernel_payload(
    generator: KernelGenerator,
    node: Mapping[str, Any],
    target: str,
) -> Dict[str, Any]:
    """Render a Triton kernel tailored to the bottleneck node."""

    template_name = _pick_template(node)
    template = generator.describe_template(template_name)
    parameters = dict(template.defaults)
    parameters.update(_tune_parameters(template_name, parameters, target, node))

    request: Dict[str, Any] = {
        "template": template_name,
        "parameters": parameters,
    }
    result = generator.generate(request, validate=False)
    source_text = Path(result.module_path).read_text()

    return {
        "template": template_name,
        "module_name": Path(result.module_path).stem,
        "module_path": str(result.module_path),
        "parameters": dict(parameters),
        "source": source_text,
    }


def _pick_template(node: Mapping[str, Any]) -> str:
    """Infer an appropriate kernel template for ``node``."""

    op = str(node.get("op") or node.get("target") or "").lower()
    name = str(node.get("name") or "").lower()
    candidates = (op, name)

    if any(token in op for token in ("matmul", "mm", "bmm", "linear", "gemm")):
        return "matrix_multiply"
    if any("layernorm" in token or "layer_norm" in token for token in candidates):
        return "layer_norm"
    if any(term in op for term in ("norm", "normalize")):
        return "layer_norm"
    return "vector_add"


def _tune_parameters(
    template_name: str,
    defaults: Mapping[str, Any],
    target: str,
    node: Mapping[str, Any],
) -> Dict[str, Any]:
    """Derive template parameters based on hardware ``target`` and node shape."""

    tuned: Dict[str, Any] = {}
    target_key = target.lower()

    shape = node.get("shape")
    dims = []
    if isinstance(shape, Sequence):
        for dim in shape:
            try:
                dims.append(int(dim))
            except (TypeError, ValueError):
                continue

    max_dim = max(dims) if dims else None

    if template_name == "matrix_multiply":
        if target_key.startswith("a100"):
            tuned["BLOCK_M"] = 128
            tuned["BLOCK_N"] = 128
            tuned["BLOCK_K"] = 64
            tuned["GROUP_M"] = 8
        elif target_key.startswith("h100"):
            tuned["BLOCK_M"] = 128
            tuned["BLOCK_N"] = 128
            tuned["BLOCK_K"] = 64
            tuned["GROUP_M"] = 6
        elif target_key.startswith("rtx"):
            tuned["BLOCK_M"] = 64
            tuned["BLOCK_N"] = 64
            tuned["BLOCK_K"] = 32
            tuned["GROUP_M"] = 4
        else:
            tuned["BLOCK_M"] = int(defaults.get("BLOCK_M", 128))
            tuned["BLOCK_N"] = int(defaults.get("BLOCK_N", 128))
            tuned["BLOCK_K"] = int(defaults.get("BLOCK_K", 32))
            tuned["GROUP_M"] = int(defaults.get("GROUP_M", 8))
        if max_dim is not None:
            tuned["BLOCK_M"] = min(tuned["BLOCK_M"], max_dim)
            tuned["BLOCK_N"] = min(tuned["BLOCK_N"], max_dim)
    elif template_name == "layer_norm":
        base = int(defaults.get("BLOCK_SIZE", 128))
        if target_key.startswith("a100") or target_key.startswith("h100"):
            tuned["BLOCK_SIZE"] = min(256, max(base, 192))
        else:
            tuned["BLOCK_SIZE"] = base
        if max_dim is not None:
            tuned["BLOCK_SIZE"] = min(tuned["BLOCK_SIZE"], max(64, max_dim))
    else:  # vector_add or fallback
        base = int(defaults.get("BLOCK_SIZE", 128))
        if target_key.startswith("a100"):
            tuned["BLOCK_SIZE"] = min(512, max(base, 256))
        elif target_key.startswith("h100"):
            tuned["BLOCK_SIZE"] = min(512, max(base, 288))
        else:
            tuned["BLOCK_SIZE"] = base
        if max_dim is not None:
            tuned["BLOCK_SIZE"] = min(tuned["BLOCK_SIZE"], max(32, max_dim))

    return tuned


def _build_patch_plan(
    *,
    focus_node: Mapping[str, Any],
    target: str,
    telemetry_summary: Mapping[str, Any],
    kernel_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Create patch metadata and human-readable instructions."""

    baseline_latency = _node_latency(focus_node)
    if baseline_latency is None:
        baseline_latency = float(telemetry_summary.get("max_latency_ms") or 0.0)

    expected_speedup_pct = _estimate_speedup_pct(focus_node, target)
    expected_latency = baseline_latency * (1.0 - expected_speedup_pct / 100.0)

    telemetry_hint = telemetry_summary.get("bottleneck_event") or {}

    instructions = [
        {
            "order": 1,
            "title": "Load generated kernel",
            "description": (
                f"Import `{kernel_payload['module_name']}` from {kernel_payload['module_path']} "
                "and retain the `run_kernel` entrypoint."
            ),
        },
        {
            "order": 2,
            "title": "Construct FX patch",
            "description": (
                "Create an `FXNodePatch` targeting "
                f"`{focus_node.get('target') or focus_node.get('op')}` and assign "
                f"the imported `run_kernel` as the replacement."
            ),
        },
        {
            "order": 3,
            "title": "Apply runtime patcher",
            "description": (
                "Invoke `RuntimePatcher().patch(model, fx_patches=[patch], copy_module=True)` "
                "to obtain an instrumented module preserving the baseline copy."
            ),
        },
        {
            "order": 4,
            "title": "Validate numerical parity",
            "description": (
                "Run representative inputs through the patched module and verify "
                "outputs match the baseline within tolerance before deployment."
            ),
        },
    ]

    return {
        "strategy": "fx",
        "instructions": instructions,
        "metrics": {
            "baseline_latency_ms": round(baseline_latency, 6),
            "expected_latency_ms": round(expected_latency, 6),
            "expected_speedup_pct": round(expected_speedup_pct, 3),
        },
        "telemetry": {
            "event_name": telemetry_hint.get("name"),
            "event_latency_ms": _first_float(
                telemetry_hint, ("cuda_time_total", "cuda_time_ms", "cpu_time_total", "cpu_time_ms")
            ),
        },
    }


def _estimate_speedup_pct(node: Mapping[str, Any], target: str) -> float:
    """Heuristic speed-up estimate based on op category and hardware."""

    op = str(node.get("op") or node.get("target") or "").lower()
    base = 18.0

    if any(token in op for token in ("matmul", "mm", "gemm", "linear")):
        base = 32.0
    elif "layernorm" in op or "layer_norm" in op or "norm" in op:
        base = 22.0
    elif "attention" in op or "attn" in op:
        base = 27.0

    target_key = target.lower()
    if target_key.startswith("a100"):
        base *= 1.1
    elif target_key.startswith("h100"):
        base *= 1.2
    elif target_key.startswith("rtx"):
        base *= 0.85

    return max(5.0, min(60.0, base))


def _craft_optimized_graph(
    nodes: Sequence[Mapping[str, Any]],
    focus_node: Mapping[str, Any],
    patch_plan: Mapping[str, Any],
    target: str,
) -> List[Dict[str, Any]]:
    """Attach optimization annotations to the graph."""

    focus_name = str(focus_node.get("name"))
    annotations = {
        "status": "optimized",
        "strategy": patch_plan.get("strategy", "fx"),
        "target": target,
        "baseline_latency_ms": patch_plan["metrics"]["baseline_latency_ms"],
        "expected_latency_ms": patch_plan["metrics"]["expected_latency_ms"],
        "expected_speedup_pct": patch_plan["metrics"]["expected_speedup_pct"],
    }

    optimized: List[Dict[str, Any]] = []
    for node in nodes:
        entry = copy.deepcopy(dict(node))
        entry_annotations: MutableMapping[str, Any]
        if str(node.get("name")) == focus_name:
            entry_annotations = dict(entry.get("annotations") or {})
            entry_annotations.update(annotations)
            entry["annotations"] = entry_annotations
        else:
            entry_annotations = dict(entry.get("annotations") or {})
            entry_annotations.setdefault("status", "baseline")
            entry_annotations.setdefault("target", target)
            entry["annotations"] = entry_annotations
        optimized.append(entry)
    return optimized
