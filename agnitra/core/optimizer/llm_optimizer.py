"""LLM-powered optimizer prompt engine.

This module builds prompts for non-interactive Codex usage (``codex exec``) or
direct Responses API calls and parses the outputs into structured kernel tuning
suggestions. It accepts telemetry summaries alongside an IR graph description
of the profiled model and returns a deterministic JSON payload so downstream
components may consume the LLM output without tightly coupling to the model
vendor response schema.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

LOGGER = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are an elite GPU kernel optimization architect embedded in an automated"
    " compiler pipeline. Your objective is to maximize CUDA latency reductions"
    " while preserving exact numerical correctness and respecting hardware"
    " limits. Study the telemetry to pinpoint the dominant bottleneck (op,"
    " shape, latency, memory footprint) and the IR graph to understand data-flow"
    " dependencies, tensor strides, and launch topology. Synthesize aggressive"
    " yet realistic improvements using techniques such as block-size tuning,"
    " warp tiling, shared-memory staging, double buffering, register reuse,"
    " vectorized loads, warp-level primitives, and occupancy balancing. Estimate"
    " the achievable latency after optimization, ensuring it improves upon the"
    " measured baseline. Respond with a single JSON object containing only the"
    " keys block_size (int), tile_shape (list of two ints), unroll_factor (int),"
    " target_latency_ms (float for desired target), expected_latency_ms (float"
    " for your forecasted result), and rationale (concise sentence explaining"
    " the performance win). Do not include Markdown, commentary, or additional"
    " fields."
)


@dataclass
class LLMOptimizerConfig:
    """Configuration for :class:`LLMOptimizer`."""

    model: str = "gpt-5-mini"
    fallback_model: Optional[str] = "gpt-4.1-mini"
    max_output_tokens: int = 400
    temperature: float = 0.0
    top_p: float = 0.9
    fallback_latency_reduction_pct: float = 0.2
    backend: str = field(
        default_factory=lambda: os.getenv("AGNITRA_LLM_BACKEND", "responses")
    )
    codex_cli_path: str = field(
        default_factory=lambda: os.getenv("AGNITRA_CODEX_PATH", "codex")
    )
    codex_cli_args: Sequence[str] = field(default_factory=lambda: ())
    # Ollama local model settings (used when backend="ollama")
    ollama_base_url: str = field(
        default_factory=lambda: os.getenv("AGNITRA_OLLAMA_URL", "http://localhost:11434")
    )
    ollama_model: str = field(
        default_factory=lambda: os.getenv("AGNITRA_OLLAMA_MODEL", "llama3")
    )


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


@dataclass
class ModelSuggestionResult:
    """Holds the outcome of querying a single model or fallback path."""

    model: str
    status: str
    suggestion: Optional[LLMOptimizationSuggestion] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None

    def to_public_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": self.model, "status": self.status}
        if self.suggestion:
            payload["suggestion"] = self.suggestion.to_dict()
        if self.error:
            payload["error"] = self.error
        if self.raw_text:
            payload["raw_text"] = self.raw_text
        return payload


class LLMOptimizer:
    """Prompt engine that queries an LLM for kernel tuning suggestions."""

    def __init__(
        self,
        client: Any | None = None,
        config: Optional[LLMOptimizerConfig] = None,
    ) -> None:
        self._client = client
        self._config = config or LLMOptimizerConfig()
        self._resolved_codex_cli = self._resolve_codex_executable(self._config.codex_cli_path)
        self._codex_cli_available = bool(self._resolved_codex_cli)
        self._auto_selected_backend = False
        requested_backend = (self._config.backend or "responses").lower()
        if requested_backend == "auto":
            if self._client is not None:
                self._backend = "responses"
            elif self._codex_cli_available:
                self._backend = "codex_cli"
                self._auto_selected_backend = True
                LOGGER.info(
                    "LLM optimizer auto-selected Codex CLI backend because no client was provided."
                )
            elif self._probe_ollama(self._config.ollama_base_url):
                self._backend = "ollama"
                self._auto_selected_backend = True
                LOGGER.info(
                    "LLM optimizer auto-selected Ollama backend (local model detected at %s).",
                    self._config.ollama_base_url,
                )
            else:
                self._backend = "responses"
        else:
            self._backend = requested_backend
        self.last_messages: Optional[Sequence[Dict[str, Any]]] = None
        self.last_response_text: Optional[str] = None
        self.last_suggestion: Optional[LLMOptimizationSuggestion] = None
        self.last_model_name: Optional[str] = None
        self.last_results: List[ModelSuggestionResult] = []

    def optimize(
        self,
        graph: Any,
        telemetry: Any | None = None,
        target_latency_ms: Optional[float] = None,
    ) -> str:
        """Generate tuned kernel parameters for the provided IR + telemetry."""

        baseline_event = _select_bottleneck_event(telemetry)
        baseline_latency = _extract_latency(baseline_event)
        if baseline_latency is not None:
            self._emit_checkpoint(
                f"Before optimization checkpoint: bottleneck latency {baseline_latency:.3f} ms"
            )
        else:
            self._emit_checkpoint("Before optimization checkpoint: no latency baseline available")
        candidate_models = self._candidate_models()
        if candidate_models:
            self._emit_checkpoint(
                "Preferred model order: " + ", ".join(candidate_models)
            )
        if self._auto_selected_backend:
            self._emit_checkpoint(
                "Auto-selected Codex CLI backend (non-interactive mode detected)."
            )
        messages = self._build_messages(graph, telemetry, target_latency_ms)
        self.last_messages = messages
        results = self._collect_model_suggestions(
            messages,
            telemetry,
            target_latency_ms,
            candidate_models,
        )
        has_valid_payload = any(
            self._has_structured_payload(result.suggestion) for result in results
        )
        if not has_valid_payload:
            self._emit_checkpoint(
                "LLM attempts yielded no structured suggestions; returning heuristic fallback"
            )
            fallback_text = self._fallback_suggestion_text(telemetry, target_latency_ms)
            fallback_suggestion = self._parse_suggestion(fallback_text)
            results.append(
                ModelSuggestionResult(
                    model="heuristic-fallback",
                    status="fallback",
                    suggestion=fallback_suggestion,
                    raw_text=fallback_text,
                )
            )
        self.last_results = results
        best_result = self._select_best_result(results, baseline_latency)
        if best_result:
            self.last_suggestion = best_result.suggestion
            self.last_model_name = best_result.model
            self.last_response_text = best_result.raw_text
            self._log_suggestion(best_result.suggestion)
            self._emit_checkpoint(f"Best model selected: {best_result.model}")
        else:
            self.last_suggestion = None
            self.last_model_name = None
            self.last_response_text = None
            self._emit_checkpoint("No structured suggestion available; fallback data only")
        self._emit_summary(baseline_latency, results)
        report = self._build_report(
            baseline_latency,
            baseline_event,
            results,
            best_result,
        )
        return json.dumps(report, sort_keys=True)

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

    def _collect_model_suggestions(
        self,
        messages: Sequence[Dict[str, Any]],
        telemetry: Any | None,
        target_latency_ms: Optional[float],
        models: Optional[Sequence[str]] = None,
    ) -> List[ModelSuggestionResult]:
        evaluated: List[ModelSuggestionResult] = []
        model_list = list(models or [])
        if not model_list:
            return evaluated
        if self._backend == "responses" and self._client is None:
            message = "LLM client not configured; skipping direct model calls"
            LOGGER.debug(message)
            self._emit_checkpoint(message)
            for model_name in model_list:
                evaluated.append(
                    ModelSuggestionResult(
                        model=model_name,
                        status="skipped",
                        error="client not configured",
                    )
                )
            return evaluated
        if self._backend == "ollama" and not self._probe_ollama(self._config.ollama_base_url):
            message = f"Ollama not reachable at {self._config.ollama_base_url}; skipping"
            LOGGER.warning(message)
            self._emit_checkpoint(message)
            for model_name in model_list:
                evaluated.append(
                    ModelSuggestionResult(
                        model=model_name,
                        status="skipped",
                        error="ollama not reachable",
                    )
                )
            return evaluated
        if self._backend == "codex_cli" and not self._codex_cli_available:
            message = "Codex CLI backend requested but executable not found"
            LOGGER.warning(message)
            self._emit_checkpoint(message)
            for model_name in model_list:
                evaluated.append(
                    ModelSuggestionResult(
                        model=model_name,
                        status="error",
                        error="codex CLI unavailable",
                    )
                )
            return evaluated

        for index, model_name in enumerate(model_list):
            try:
                LOGGER.debug("LLM request using model %s via %s backend", model_name, self._backend)
                self._emit_checkpoint(f"Attempting optimization with model '{model_name}'")
                raw_text = self._dispatch_to_model(messages, model_name)
                suggestion = self._parse_suggestion(raw_text)
                status = "ok" if self._has_structured_payload(suggestion) else "empty"
                evaluated.append(
                    ModelSuggestionResult(
                        model=model_name,
                        status=status,
                        suggestion=suggestion,
                        raw_text=raw_text,
                    )
                )
                if index > 0 and status == "ok":
                    LOGGER.info("LLM fallback model %s succeeded", model_name)
                self._emit_checkpoint(
                    f"Model '{model_name}' completed with status '{status}'"
                )
            except Exception as exc:  # pragma: no cover - network/process failures
                LOGGER.warning(
                    "LLM request failed for model %s (%s)",
                    model_name,
                    exc,
                )
                self._emit_checkpoint(f"Model '{model_name}' failed: {exc}")
                evaluated.append(
                    ModelSuggestionResult(
                        model=model_name,
                        status="error",
                        error=str(exc),
                    )
                )
        return evaluated

    def _dispatch_to_model(
        self,
        messages: Sequence[Dict[str, Any]],
        model_name: str,
    ) -> str:
        if self._backend == "codex_cli":
            return self._invoke_codex_cli(messages, model_name)
        if self._backend == "ollama":
            return self._invoke_ollama(messages)
        return self._invoke_responses_api(messages, model_name)

    def _invoke_responses_api(
        self,
        messages: Sequence[Dict[str, Any]],
        model_name: str,
    ) -> str:
        client = self._client
        if client is None:
            raise RuntimeError("LLM client not configured for responses backend")
        model_key = str(model_name).strip().lower()

        def _supports_sampling_controls() -> bool:
            # The canonical gpt-5 model refuses decoding parameters such as temperature,
            # top_p, and max_output_tokens. Guard against that specific model name while
            # still allowing sibling variants (e.g. gpt-5-mini) to receive the knobs.
            return model_key not in {"gpt-5"}

        payload: Dict[str, Any] = {
            "model": model_name,
            "input": messages,
            "store": False,
        }
        supports_sampling = _supports_sampling_controls()
        # Respect configured decoding controls where supported by the target model.
        if self._config.max_output_tokens is not None and supports_sampling:
            payload["max_output_tokens"] = int(self._config.max_output_tokens)
        if (
            self._config.temperature is not None
            and supports_sampling
        ):
            payload["temperature"] = float(self._config.temperature)
        if (
            self._config.top_p is not None
            and supports_sampling
        ):
            payload["top_p"] = float(self._config.top_p)
        response = client.responses.create(**payload)
        return _extract_text(response)

    def _invoke_codex_cli(
        self,
        messages: Sequence[Dict[str, Any]],
        model_name: str,
    ) -> str:
        if not self._codex_cli_available:
            raise RuntimeError("Codex CLI executable not available")
        prompt = self._render_cli_prompt(messages)
        executable = self._resolved_codex_cli or self._config.codex_cli_path
        command: List[str] = [executable, "exec", "--full-auto"]
        if model_name:
            command.extend(["--model", model_name])
        for extra in self._config.codex_cli_args:
            command.append(str(extra))
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("codex CLI not found") from exc
        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        if completed.returncode != 0:
            message = f"codex exec failed with exit code {completed.returncode}"
            if stderr:
                message += f": {stderr}"
            raise RuntimeError(message)
        if stderr:
            LOGGER.debug("codex exec stderr: %s", stderr)
        return stdout

    def _invoke_ollama(self, messages: Sequence[Dict[str, Any]]) -> str:
        """Send a chat completion request to a local Ollama server."""
        try:
            import httpx  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for the Ollama backend: pip install httpx"
            ) from exc

        prompt = self._render_cli_prompt(messages)
        base_url = self._config.ollama_base_url.rstrip("/")
        model = self._config.ollama_model
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            response = httpx.post(
                f"{base_url}/api/generate",
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except Exception as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    @staticmethod
    def _probe_ollama(base_url: str) -> bool:
        """Return True when an Ollama server is reachable at *base_url*."""
        try:
            import httpx  # type: ignore[import-not-found]
            response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False

    def _render_cli_prompt(self, messages: Sequence[Dict[str, Any]]) -> str:
        segments: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            prefix = role.upper()
            text_parts: List[str] = []
            for item in message.get("content", []) or []:
                text = item.get("text") if isinstance(item, Mapping) else None
                if text:
                    text_parts.append(str(text))
            if text_parts:
                segments.append(f"[{prefix}]\n" + "\n".join(text_parts))
        segments.append(
            "Please respond with a single JSON object containing the requested keys only."
        )
        return "\n\n".join(segments)

    def _resolve_codex_executable(self, candidate: str | None) -> Optional[str]:
        if not candidate:
            return None
        candidate = str(candidate)
        if os.path.isabs(candidate):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
            return None
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
        return None

    def _has_structured_payload(
        self, suggestion: Optional[LLMOptimizationSuggestion]
    ) -> bool:
        if suggestion is None:
            return False
        payload = suggestion.to_dict()
        return any(key != "source" for key in payload)

    def _select_best_result(
        self,
        results: Sequence[ModelSuggestionResult],
        baseline_latency: Optional[float],
    ) -> Optional[ModelSuggestionResult]:
        candidates = [
            result
            for result in results
            if self._has_structured_payload(result.suggestion)
        ]
        if not candidates:
            return None

        def _score(result: ModelSuggestionResult) -> float:
            suggestion = result.suggestion
            if suggestion is None:
                return float("inf")
            if suggestion.expected_latency_ms is not None:
                return float(suggestion.expected_latency_ms)
            if suggestion.target_latency_ms is not None:
                return float(suggestion.target_latency_ms)
            if baseline_latency is not None:
                return float(baseline_latency)
            return float("inf")

        return min(candidates, key=_score)

    def _emit_summary(
        self,
        baseline_latency: Optional[float],
        results: Sequence[ModelSuggestionResult],
    ) -> None:
        lines: List[str] = ["Detailed check:"]
        if baseline_latency is not None:
            lines.append(f"  Baseline latency: {baseline_latency:.3f} ms")
        else:
            lines.append("  Baseline latency: unavailable")
        lines.append("  Before/After assessments:")
        for result in results:
            suggestion = result.suggestion
            if self._has_structured_payload(suggestion):
                lines.append(
                    f"    - {result.model}: {self._format_before_after(baseline_latency, suggestion)}"
                )
                summary = self._summarise_suggestion(suggestion)
                lines.append(f"      Suggestion: {summary}")
                issues = self._validate_suggestion(suggestion)
                if issues:
                    lines.append("      Checks: " + ", ".join(issues))
                else:
                    lines.append("      Checks: all mandatory fields present")
            else:
                extra = f" ({result.status})" if result.status else ""
                lines.append(f"    - {result.model}: no suggestion{extra}")
                if result.error:
                    lines.append(f"      Reason: {result.error}")
        summary = "\n".join(lines)
        self._emit_checkpoint("Before/After analysis:\n" + summary)

    def _validate_suggestion(
        self, suggestion: LLMOptimizationSuggestion
    ) -> List[str]:
        issues: List[str] = []
        if suggestion.block_size is None:
            issues.append("block_size missing")
        elif suggestion.block_size <= 0:
            issues.append("block_size must be > 0")
        if suggestion.tile_shape is None or len(suggestion.tile_shape) != 2:
            issues.append("tile_shape missing")
        else:
            if any(dim <= 0 for dim in suggestion.tile_shape):
                issues.append("tile_shape must be positive")
        if suggestion.unroll_factor is None:
            issues.append("unroll_factor missing")
        elif suggestion.unroll_factor <= 0:
            issues.append("unroll_factor must be > 0")
        if suggestion.expected_latency_ms is None and suggestion.target_latency_ms is None:
            issues.append("latency estimates missing")
        if not suggestion.rationale:
            issues.append("rationale missing")
        return issues

    def _format_before_after(
        self,
        baseline_latency: Optional[float],
        suggestion: LLMOptimizationSuggestion,
    ) -> str:
        expected = suggestion.expected_latency_ms
        target = suggestion.target_latency_ms
        if baseline_latency is not None and expected is not None:
            improvement = baseline_latency - expected
            percent = (improvement / baseline_latency * 100.0) if baseline_latency else 0.0
            return (
                f"baseline {baseline_latency:.3f} ms -> expected {expected:.3f} ms "
                f"(improvement {improvement:.3f} ms, {percent:.1f}%)"
            )
        if baseline_latency is not None and target is not None:
            improvement = baseline_latency - target
            percent = (improvement / baseline_latency * 100.0) if baseline_latency else 0.0
            return (
                f"baseline {baseline_latency:.3f} ms -> target {target:.3f} ms "
                f"(goal improvement {improvement:.3f} ms, {percent:.1f}%)"
            )
        if expected is not None:
            return f"expected {expected:.3f} ms (baseline unavailable)"
        if target is not None:
            return f"target {target:.3f} ms (baseline unavailable)"
        return "no latency estimates provided"

    def _build_report(
        self,
        baseline_latency: Optional[float],
        baseline_event: Optional[Mapping[str, Any]],
        results: Sequence[ModelSuggestionResult],
        best_result: Optional[ModelSuggestionResult],
    ) -> Dict[str, Any]:
        models_payload: List[Dict[str, Any]] = []
        for result in results:
            entry = result.to_public_dict()
            if result.suggestion and not self._has_structured_payload(result.suggestion):
                entry.pop("suggestion", None)
            models_payload.append(entry)
        baseline_payload: Dict[str, Any] = {"latency_ms": baseline_latency}
        if baseline_event:
            baseline_payload.update(
                {
                    "op": baseline_event.get("op") or baseline_event.get("name"),
                    "shape": baseline_event.get("shape") or baseline_event.get("shapes"),
                }
            )
        report: Dict[str, Any] = {
            "baseline": baseline_payload,
            "models": models_payload,
        }
        if best_result and best_result.suggestion:
            best_payload: Dict[str, Any] = {
                "model": best_result.model,
                "suggestion": best_result.suggestion.to_dict(),
            }
            improvement = self._compute_improvement(
                baseline_latency,
                best_result.suggestion,
            )
            if improvement:
                best_payload.update(improvement)
            report["best_model"] = best_payload
        return report

    def _compute_improvement(
        self,
        baseline_latency: Optional[float],
        suggestion: LLMOptimizationSuggestion,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if suggestion.expected_latency_ms is not None:
            payload["expected_latency_ms"] = suggestion.expected_latency_ms
        if suggestion.target_latency_ms is not None:
            payload["target_latency_ms"] = suggestion.target_latency_ms
        if baseline_latency is None:
            return payload
        reference = suggestion.expected_latency_ms or suggestion.target_latency_ms
        if reference is None:
            return payload
        improvement = baseline_latency - reference
        payload["baseline_latency_ms"] = baseline_latency
        payload["improvement_ms"] = improvement
        if baseline_latency:
            payload["improvement_pct"] = improvement / baseline_latency * 100.0
        return payload

    def _candidate_models(self) -> list[str]:
        models = []
        primary = os.getenv("AGNITRA_LLM_MODEL", self._config.model)
        if primary:
            models.append(primary)
        fallback = os.getenv("AGNITRA_LLM_FALLBACK_MODEL", "") or self._config.fallback_model
        if fallback and fallback not in models:
            models.append(fallback)
        return models or [self._config.model]

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

    def _emit_checkpoint(self, message: str) -> None:
        LOGGER.info("[LLM optimizer] %s", message)
        print(f"[LLM optimizer] {message}")

    def _summarise_suggestion(self, suggestion: LLMOptimizationSuggestion) -> str:
        parts = []
        if suggestion.block_size is not None:
            parts.append(f"block_size={suggestion.block_size}")
        if suggestion.tile_shape is not None:
            parts.append(f"tile_shape={suggestion.tile_shape[0]}x{suggestion.tile_shape[1]}")
        if suggestion.unroll_factor is not None:
            parts.append(f"unroll_factor={suggestion.unroll_factor}")
        if suggestion.expected_latency_ms is not None:
            parts.append(f"expected_latency={suggestion.expected_latency_ms:.3f} ms")
        if suggestion.target_latency_ms is not None:
            parts.append(f"target_latency={suggestion.target_latency_ms:.3f} ms")
        if suggestion.rationale:
            parts.append(f"rationale={suggestion.rationale}")
        if not parts:
            return "no tuning parameters returned"
        return ", ".join(parts)

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
        if "content" in response and response["content"] is not None:
            return _extract_text(response["content"])
        if "text" in response and isinstance(response["text"], str):
            return response["text"]
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
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = text[start : end + 1]
            try:
                parsed = json.loads(snippet)
            except json.JSONDecodeError:
                return None
        else:
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
