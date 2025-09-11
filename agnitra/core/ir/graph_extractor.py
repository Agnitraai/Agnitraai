"""FX-based IR graph extractor with telemetry annotations.

This module provides a thin wrapper around ``torch.fx.symbolic_trace`` to
produce a JSON-serializable IR and loosely attach any matching telemetry
entries by name.
"""

from __future__ import annotations

from typing import Any, Dict, List

try:  # pragma: no cover - optional dependency
    import torch
    from torch.fx import symbolic_trace
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None
    symbolic_trace = None  # type: ignore


def extract_fx_ir(model: Any, telemetry: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
    """Extract a simple FX IR for ``model`` and annotate with ``telemetry``.

    Returns a list of node dicts with fields: op, target, args, kwargs, telemetry.
    """
    if torch is None or symbolic_trace is None:
        return []
    traced = symbolic_trace(model)
    tl = telemetry or []
    nodes: List[Dict[str, Any]] = []
    for node in traced.graph.nodes:
        matched = next((t for t in tl if node.target and str(node.target) in str(t.get("name", ""))), None)
        nodes.append(
            {
                "op": node.op,
                "target": str(node.target),
                "args": str(node.args),
                "kwargs": str(node.kwargs),
                "telemetry": matched,
            }
        )
    return nodes


__all__ = ["extract_fx_ir"]

