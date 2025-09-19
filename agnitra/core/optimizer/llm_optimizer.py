"""LLM-powered optimizer prompt engine.

This module builds prompts for non-interactive Codex usage (``codex exec``)
and parses the responses into structured kernel tuning suggestions. It accepts
telemetry summaries alongside an IR graph description of the profiled model and
returns a deterministic JSON payload so downstream components may consume the
LLM output without tightly coupling to the model vendor response schema.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an expert GPU kernel tuner. Given profiling telemetry and an IR "
    "graph, you propose block size, tiling strategy, unroll factors and other "
    "parameters that reduce CUDA latency while preserving correctness. "
    "Respond with JSON using keys: block_size, tile_shape (list of two ints), "
    "unroll_factor, target_latency_ms, expected_latency_ms, rationale."
)


@dataclass
class LLMOptimizerConfig:
    """Configuration for :class:`LLMOptimizer`."""

    model: str = "o4-mini"
    max_output_tokens: int = 400
    temperature: float = 0.0
    top_p: float = 0.9
    fallback_latency_reduction_pct: float = 0.2


@dataclass
class LLMOptimizationSuggestion:
    """Structured representation of tuning suggestions."""

    block_size: Optional[int] = None
    tile_shape: Optional[tuple[int, int]] = None
    unroll_factor: Optional[int] = None
    target_latency_ms: Optional[float] = None
    expected_latency_ms: Optional[float] = None
    rationale: Optional[str] = None
    source: str = "llm"
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "block_size": self.block_size,
            "tile_shape": list(self.tile_shape) if self.tile_shape else None,
            "unroll_factor": self.unroll_factor,
            "target_latency_ms": self.target_latency_ms,
            "expected_latency_ms": self.expected_latency_ms,
            "rationale": self.rationale,
            "source": self.source,
        }
        return {k: v for k, v in data.items() if v is not None}

    def as_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class LLMOptimizer:
    """Prompt engine that queries an LLM for kernel tuning suggestions."""

    def __init__(
        self,
        client: Any | None = None,
        config: Optional[LLMOptimizerConfig] = None,
    ) -> None:
        self._client = client
        self._config = config or LLMOptimizerConfig()
        self.last_messages: Optional[Sequence[Dict[str, Any]]] = None
        self.last_response_text: Optional[str] = None
        self.last_suggestion: Optional[LLMOptimizationSuggestion] = None

    def optimize(
        self,
        graph: Any,
        telemetry: Any | None = None,
        target_latency_ms: Optional[float] = None,
    ) -> str:
        """Generate tuned kernel parameters for the provided IR + telemetry."""

        messages = self._build_messages(graph, telemetry, target_latency_ms)
        self.last_messages = messages
        response_text = self._call_model(messages, telemetry, target_latency_ms)
        self.last_response_text = response_text
        suggestion = self._parse_suggestion(response_text)
        self.last_suggestion = suggestion
        self._log_suggestion(suggestion)
        return suggestion.as_json()

    def _build_messages(
        self,
        graph: Any,
        telemetry: Any | None,
        target_latency_ms: Optional[float],
    ) -> Sequence[Dict[str, Any]]:
        graph_snippet = self._serialise(graph)
        telemetry_snippet = self._summarise_telemetry(telemetry)
        prompt = (
            "Telemetry summary:\n"
            f"{telemetry_snippet}\n\n"
            "IR graph snippet:\n"
            f"{graph_snippet}\n\n"
            "Please recommend CUDA kernel tuning parameters that reduce latency"
        )
        if target_latency_ms is not None:
            prompt += f" below {target_latency_ms:.3f} ms."
        else:
            prompt += "."
        prompt += (
            " Provide JSON with keys block_size, tile_shape, unroll_factor, "
            "target_latency_ms, expected_latency_ms, rationale."
        )
        messages = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": _DEFAULT_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            },
        ]
        return messages

    def _call_model(
        self,
        messages: Sequence[Dict[str, Any]],
        telemetry: Any | None,
        target_latency_ms: Optional[float],
    ) -> str:
        client = self._client
        if client is None:
            LOGGER.debug("LLM client not configured; using heuristic fallback")
            return self._fallback_suggestion_text(telemetry, target_latency_ms)
        try:
            response = client.responses.create(
                model=self._config.model,
                input=messages,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                max_output_tokens=self._config.max_output_tokens,
                store=False,
            )
        except Exception as exc:  # pragma: no cover - network failures
            LOGGER.warning("LLM request failed (%s); using fallback", exc)
            return self._fallback_suggestion_text(telemetry, target_latency_ms)
        return _extract_text(response)

    def _fallback_suggestion_text(
        self,
        telemetry: Any | None,
        target_latency_ms: Optional[float],
    ) -> str:
        event = _select_bottleneck_event(telemetry)
        baseline = _extract_latency(event)
        if baseline is None:
            baseline = 10.2
        reduction = baseline * (1.0 - self._config.fallback_latency_reduction_pct)
        target = target_latency_ms or reduction
        suggestion = {
            "block_size": 128,
            "tile_shape": [64, 64],
            "unroll_factor": 2,
            "target_latency_ms": target,
            "expected_latency_ms": max(target - 0.4, target * 0.9),
            "source": "fallback",
        }
        if event:
            shape = event.get("shape") or event.get("shapes")
            op_name = event.get("op") or event.get("name")
            suggestion["rationale"] = (
                f"Heuristic fallback for {op_name or 'kernel'} "
                f"with shape {shape or '[1024, 1024]'}."
            )
        else:
            suggestion["rationale"] = "Heuristic fallback suggestion without telemetry context."
        return json.dumps(suggestion)

    def _parse_suggestion(self, text: str) -> LLMOptimizationSuggestion:
        cleaned = _strip_code_fences(text.strip())
        if not cleaned:
            return LLMOptimizationSuggestion(raw_text=text or None, source="empty")
        data = _parse_json_payload(cleaned)
        if data:
            return LLMOptimizationSuggestion(
                block_size=_coerce_int(data.get("block_size")),
                tile_shape=_coerce_tile(data.get("tile_shape")),
                unroll_factor=_coerce_int(data.get("unroll_factor")),
                target_latency_ms=_coerce_float(data.get("target_latency_ms")),
                expected_latency_ms=_coerce_float(data.get("expected_latency_ms")),
                rationale=_coerce_str(data.get("rationale")),
                source=_coerce_str(data.get("source")) or "llm",
                raw_text=cleaned,
            )
        parsed = _parse_key_value_text(cleaned)
        return LLMOptimizationSuggestion(
            block_size=parsed.get("block_size"),
            tile_shape=parsed.get("tile_shape"),
            unroll_factor=parsed.get("unroll_factor"),
            target_latency_ms=parsed.get("target_latency_ms"),
            expected_latency_ms=parsed.get("expected_latency_ms"),
            rationale=parsed.get("rationale"),
            source=parsed.get("source", "llm"),
            raw_text=cleaned,
        )

    def _log_suggestion(self, suggestion: LLMOptimizationSuggestion) -> None:
        LOGGER.info(
            "LLM suggestion source=%s block=%s tile=%s unroll=%s target=%.3f expected=%.3f",
            suggestion.source,
            suggestion.block_size,
            suggestion.tile_shape,
            suggestion.unroll_factor,
            suggestion.target_latency_ms or -1.0,
            suggestion.expected_latency_ms or -1.0,
        )
        if suggestion.rationale:
            LOGGER.debug("LLM rationale: %s", suggestion.rationale)

    @staticmethod
    def _serialise(payload: Any, max_chars: int = 2000) -> str:
        if payload is None:
            return "<absent>"
        if isinstance(payload, str):
            text = payload
        else:
            try:
                text = json.dumps(payload, indent=2, sort_keys=True)
            except (TypeError, ValueError):
                text = repr(payload)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    @staticmethod
    def _summarise_telemetry(telemetry: Any | None) -> str:
        if telemetry is None:
            return "No telemetry provided."
        try:
            events = list(_iter_events(telemetry))
        except Exception:
            return LLMOptimizer._serialise(telemetry)
        if not events:
            return "No telemetry events detected."
        top = max(events, key=_score_event)
        snippet = {
            "bottleneck_op": top.get("op") or top.get("name"),
            "shape": top.get("shape"),
            "cuda_time_ms": _extract_latency(top),
            "associated_events": len(events),
        }
        return json.dumps(snippet, indent=2, sort_keys=True)


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if response is None:
        return ""
    if isinstance(response, Mapping):
        maybe_output = response.get("output") or response.get("choices")
        if maybe_output:
            return _extract_text(maybe_output)
        return json.dumps(response)
    if isinstance(response, Sequence):
        parts = [
            _extract_text(item)
            for item in response
            if not isinstance(item, (str, bytes)) or item
        ]
        return "".join(parts)
    try:
        output = getattr(response, "output", None)
        if output is not None:
            return _extract_text(output)
        choices = getattr(response, "choices", None)
        if choices is not None:
            return _extract_text(choices)
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return text
    except Exception:  # pragma: no cover - defensive
        pass
    return str(response)


def _strip_code_fences(text: str) -> str:
    fenced = text.strip()
    if fenced.startswith("```") and fenced.endswith("```"):
        fenced_lines = fenced.splitlines()
        if len(fenced_lines) >= 2:
            fenced = "\n".join(fenced_lines[1:-1])
    return fenced.strip()


def _parse_json_payload(text: str) -> Dict[str, Any] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        if not parsed:
            return None
        first = parsed[0]
        if isinstance(first, Mapping):
            return dict(first)
        return None
    if isinstance(parsed, Mapping):
        return dict(parsed)
    return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _coerce_tile(value: Any) -> Optional[tuple[int, int]]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.search(r"(\d+)[xX](\d+)", value)
        if match:
            return (int(match.group(1)), int(match.group(2)))
    return None


def _parse_key_value_text(text: str) -> Dict[str, Any]:
    block = _find_int(text, r"block(?:_|\s*)size\D*(\d+)")
    tile = _find_tile(text)
    unroll = _find_int(text, r"unroll(?:_|\s*)factor\D*(\d+)")
    target = _find_float(text, r"target[^\d]*(\d+(?:\.\d+)?)\s*ms")
    expected = _find_float(text, r"expected[^\d]*(\d+(?:\.\d+)?)\s*ms")
    if expected is None:
        expected = _find_float(text, r"latency[^\d]*(\d+(?:\.\d+)?)\s*ms")
    rationale = None
    match = re.search(r"rationale\s*[:\-]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        rationale = match.group(1).strip()
    return {
        "block_size": block,
        "tile_shape": tile,
        "unroll_factor": unroll,
        "target_latency_ms": target,
        "expected_latency_ms": expected,
        "rationale": rationale,
        "source": "llm",
    }


def _find_int(text: str, pattern: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def _find_float(text: str, pattern: str) -> Optional[float]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, TypeError):
            return None
    return None


def _find_tile(text: str) -> Optional[tuple[int, int]]:
    match = re.search(r"tile(?:_|\s*)(?:shape|size)[^\d]*(\d+)[^\d]+(\d+)", text, flags=re.IGNORECASE)
    if match:
        try:
            return (int(match.group(1)), int(match.group(2)))
        except (ValueError, TypeError):
            return None
    return None


def _iter_events(telemetry: Any) -> Iterable[Mapping[str, Any]]:
    if telemetry is None:
        return []
    if isinstance(telemetry, Mapping):
        if "events" in telemetry and isinstance(telemetry["events"], Sequence):
            return [evt for evt in telemetry["events"] if isinstance(evt, Mapping)]
        if "bottlenecks" in telemetry and isinstance(telemetry["bottlenecks"], Sequence):
            return [evt for evt in telemetry["bottlenecks"] if isinstance(evt, Mapping)]
    if isinstance(telemetry, Sequence):
        return [evt for evt in telemetry if isinstance(evt, Mapping)]
    return []


def _score_event(event: Mapping[str, Any]) -> float:
    latency = _extract_latency(event)
    if latency is None:
        return 0.0
    return float(latency)


def _extract_latency(event: Mapping[str, Any] | None) -> Optional[float]:
    if not event:
        return None
    for key in ("cuda_time_ms", "cuda_time", "latency_ms", "time_ms"):
        if key in event and event[key] is not None:
            try:
                return float(event[key])
            except (TypeError, ValueError):
                continue
    if "cuda_time_total" in event:
        try:
            return float(event["cuda_time_total"]) / 1_000_000.0
        except (TypeError, ValueError):
            pass
    if "cuda_time_avg" in event:
        try:
            return float(event["cuda_time_avg"]) / 1_000_000.0
        except (TypeError, ValueError):
            pass
    return None


def _select_bottleneck_event(telemetry: Any | None) -> Optional[Mapping[str, Any]]:
    try:
        events = list(_iter_events(telemetry))
    except Exception:
        return None
    if not events:
        return None
    return max(events, key=_score_event)


__all__ = [
    "LLMOptimizer",
    "LLMOptimizerConfig",
    "LLMOptimizationSuggestion",
]
