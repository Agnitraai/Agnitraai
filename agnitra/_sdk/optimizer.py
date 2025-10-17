from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.profiler import ProfilerActivity, profile, record_function

from .deps import require_openai, require_sb3
from agnitra.core.optimizer import (
    OpenEvolveResult,
    OpenEvolveRunner,
    PPOKernelOptimizer,
    PPOKernelOptimizerConfig,
    run_open_evolve_from_log,
    summarize_kernel_telemetry,
)
from agnitra.core.rl import CodexGuidedAgent
from agnitra.core.runtime import apply_tuning_preset

logger = logging.getLogger(__name__)


def _infer_module_device(module: nn.Module) -> Optional[torch.device]:
    """Best-effort helper to detect which device a module currently uses."""

    for accessor in ("parameters", "buffers"):
        try:
            iterator = getattr(module, accessor)()  # type: ignore[arg-type]
        except Exception:
            continue
        for item in iterator:
            if isinstance(item, torch.Tensor):
                return item.device
    return None


def collect_telemetry(model: nn.Module, input_tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """Collect basic profiler telemetry for a single model forward pass.

    The helper aligns the input tensor with the module's device before running the
    profiler to avoid device-mismatch errors when callers reuse models that have
    already been moved to GPU.
    """

    target_device = _infer_module_device(model)
    if target_device is None and isinstance(input_tensor, torch.Tensor):
        target_device = input_tensor.device
    if target_device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(input_tensor, torch.Tensor) and input_tensor.device != target_device:
        input_tensor = input_tensor.to(target_device)

    if hasattr(model, "to"):
        try:
            model = model.to(target_device)  # type: ignore[assignment]
        except Exception:
            pass

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    telemetry: List[Dict[str, Any]] = []
    with profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            _ = model(input_tensor)
    for evt in prof.key_averages():
        cpu_time_total = getattr(evt, "cpu_time_total", 0.0)
        cuda_time_total = getattr(evt, "cuda_time_total", 0.0)
        input_shapes = getattr(evt, "input_shapes", [])
        cpu_mem = getattr(evt, "self_cpu_memory_usage", 0)
        cuda_mem = getattr(evt, "self_cuda_memory_usage", 0)
        telemetry.append(
            {
                "name": evt.key,
                "cpu_time_ms": cpu_time_total / 1e6,
                "cuda_time_ms": (cuda_time_total / 1e6) if torch.cuda.is_available() else 0.0,
                "input_shape": input_shapes,
                "cpu_memory_bytes": cpu_mem,
                "cuda_memory_bytes": cuda_mem if torch.cuda.is_available() else 0,
            }
        )
    return telemetry


def _match_telemetry_entry(telemetry: List[Dict[str, Any]], *candidates: Optional[str]) -> Optional[Dict[str, Any]]:
    """Best-effort match to align telemetry entries with IR nodes."""

    lowered_candidates: List[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        lowered_candidates.append(str(candidate).lower())
    if not lowered_candidates:
        return None

    for entry in telemetry:
        name = str(entry.get("name", "")).lower()
        for candidate in lowered_candidates:
            if candidate and candidate in name:
                return entry
    return None


def _extract_ir_from_torchscript(script_module: nn.Module, telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback IR extraction for TorchScript modules without FX support."""

    graph = getattr(script_module, "inlined_graph", None) or getattr(script_module, "graph", None)
    if graph is None:
        return []

    ir_nodes: List[Dict[str, Any]] = []
    for node in graph.nodes():
        try:
            kind = node.kind()
        except Exception:
            continue
        if kind.startswith("prim::"):
            continue
        scope = ""
        try:
            scope = node.scopeName()
        except Exception:
            scope = ""
        target = f"{scope}::{kind}" if scope else kind
        inputs = []
        try:
            inputs = [inp.debugName() if hasattr(inp, "debugName") else str(inp) for inp in node.inputs()]
        except Exception:
            inputs = []
        telemetry_match = _match_telemetry_entry(
            telemetry,
            kind,
            target.split("::")[-1],
            scope.split("/")[-1] if scope else None,
        )
        ir_nodes.append(
            {
                "op": kind,
                "target": target,
                "args": str(inputs),
                "kwargs": "{}",
                "telemetry": telemetry_match,
            }
        )
        if len(ir_nodes) >= 256:
            break
    return ir_nodes


def extract_ir(model: nn.Module, telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract a simple IR using torch.fx and attach telemetry."""
    try:
        traced = symbolic_trace(model)
    except Exception as exc:
        if hasattr(model, "inlined_graph") or hasattr(model, "graph"):
            logger.debug("Falling back to TorchScript graph extraction: %s", exc)
            return _extract_ir_from_torchscript(model, telemetry)
        raise

    ir_nodes: List[Dict[str, Any]] = []
    for node in traced.graph.nodes:
        matched = None
        target_text = str(node.target)
        matched = _match_telemetry_entry(telemetry, target_text.split(".")[-1], target_text)
        ir_nodes.append(
            {
                "op": node.op,
                "target": target_text,
                "args": str(node.args),
                "kwargs": str(node.kwargs),
                "telemetry": matched,
            }
        )
    return ir_nodes


def request_kernel_suggestions(
    telemetry: List[Dict[str, Any]],
    ir_nodes: List[Dict[str, Any]],
    client: Optional[Any] = None,
    model_name: str = "gpt-5-codex",
) -> Optional[str]:
    """Call an LLM to request kernel suggestions. Returns text or ``None``."""
    if client is None:
        try:
            OpenAI = require_openai()
            client = OpenAI()
        except Exception as exc:  # pragma: no cover - best effort
            logger.info("%s", exc)
            return None
    try:
        ir_json = json.dumps(ir_nodes)
    except TypeError:
        ir_json = json.dumps([{ "op": n["op"], "target": n["target"] } for n in ir_nodes])
    system_message = {
        "role": "system",
        "content": [
            {
                "type": "input_text",
                "text": "You are an expert GPU kernel optimizer. Given telemetry and an IR graph, suggest block size, tile size and unroll factors to reduce latency.",
            }
        ],
    }
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "input_text",
                "text": f"Telemetry: {telemetry} IR graph: {ir_json} Provide optimized kernel parameters and rationale.",
            }
        ],
    }
    # Allow overriding the model via environment variable without changing callers
    import os as _os
    _model = _os.getenv("AGNITRA_LLM_MODEL", model_name)
    response = client.responses.create(model=_model, input=[system_message, user_message], store=False)
    optimized_text = ""
    try:
        for item in getattr(response, "output", []) or []:
            for entry in getattr(item, "content", []) or []:
                optimized_text += getattr(entry, "text", "") or ""
    except (AttributeError, TypeError):
        logger.info("Unexpected response schema for kernel suggestions")
    return optimized_text.strip() if optimized_text else None


def run_rl_tuning(telemetry: List[Dict[str, Any]], ir_nodes: List[Dict[str, Any]]) -> None:
    """Run the PPO-based RL optimizer (simulated environment)."""

    summary = summarize_kernel_telemetry(telemetry)
    config = PPOKernelOptimizerConfig(telemetry_summary=summary)

    env: Any = None
    PPO: Any = None
    gym: Any = None
    try:
        PPO, gym = require_sb3()
    except RuntimeError as exc:  # pragma: no cover - optional deps missing
        logger.info("RL optimizer unavailable: %s", exc)
    else:
        try:
            env = gym.make("AgnitraKernel-v0")
        except Exception as env_exc:  # pragma: no cover - gym optional
            logger.warning("Gym environment creation failed: %s", env_exc)
        else:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                agent = PPO("MlpPolicy", env, verbose=0, device=device)
                agent.learn(total_timesteps=max(1, config.total_timesteps // 10))
            except Exception as exc:  # pragma: no cover - PPO optional
                logger.info("SB3 PPO training failed: %s", exc)
            finally:
                if hasattr(env, "close"):
                    try:
                        env.close()
                    except Exception:
                        pass

    optimizer = PPOKernelOptimizer(config=config)
    try:
        result = optimizer.train()
    except RuntimeError as exc:  # pragma: no cover - optional deps missing
        logger.info("RL optimizer unavailable: %s", exc)
        return

    strategy = result.metadata.get("strategy", "ppo")
    logger.info(
        "RL optimizer (%s) tile=%s unroll=%s fuse=%s tokens/s=%.1f latency=%.2f Δ=%.2f%%",
        strategy,
        result.tile_size,
        result.unroll_factor,
        result.fuse_kernels,
        result.tokens_per_sec,
        result.latency_ms,
        result.improvement_ratio * 100.0,
    )


def run_llm_guided_rl(
    telemetry: List[Dict[str, Any]],
    ir_nodes: List[Dict[str, Any]],
    client: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Use a Codex-guided agent to propose a tuning preset, optionally evaluated with SB3.

    Returns the chosen config dict when available; otherwise returns ``None``.
    """
    try:
        agent = CodexGuidedAgent()
        cfg = agent.propose_config(telemetry, ir_nodes, client=client)
        if not cfg:
            return None
        # Optionally evaluate via SB3 (best-effort)
        chosen = agent.evaluate_with_sb3(telemetry, [cfg]) or cfg
        logger.info("LLM-guided RL preset: %s", chosen)
        return chosen
    except Exception:  # pragma: no cover - best effort
        logger.exception("LLM-guided RL failed")
        return None


def optimize_log_with_open_evolve(
    log_path: str | Path,
    *,
    runner: Optional[OpenEvolveRunner] = None,
    iterations: Optional[int] = None,
    evaluator: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    extra_config: Optional[Mapping[str, Any]] = None,
) -> Optional[OpenEvolveResult]:
    """Best-effort optimisation of a saved Agnitra log via OpenEvolve."""

    path_obj = Path(log_path)
    if not path_obj.exists():
        logger.error("OpenEvolve log file not found: %s", path_obj)
        return None
    try:
        return run_open_evolve_from_log(
            path_obj,
            runner=runner,
            iterations=iterations,
            evaluator=evaluator,
            extra_config=extra_config,
        )
    except Exception:  # pragma: no cover - optional dependency path
        logger.exception("OpenEvolve optimisation failed for %s", path_obj)
        return None


def optimize_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    client: Optional[Any] = None,
    enable_rl: bool = True,
    *,
    preset: Optional[Mapping[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    policy: Optional[Mapping[str, Any]] = None,
) -> nn.Module:
    """Run the optimization pipeline with graceful fallbacks.

    On any stage failure, the exception is logged and the baseline model is returned
    untouched.
    """
    context_map: Dict[str, Any] = context if context is not None else {}
    context_map.setdefault("rl_enabled", bool(enable_rl))
    context_map.setdefault("policy", dict(policy or {}))
    context_map["invoked_at"] = time.time()

    applied_preset: Optional[Dict[str, Any]] = None
    applied_pass_presets: List[Dict[str, Any]] = []
    if preset:
        try:
            model = apply_tuning_preset(model, dict(preset))
            applied_preset = dict(preset)
            context_map["applied_preset_source"] = "preset_override"
        except Exception:
            logger.exception("Failed to apply preset override")

    policy_pass_presets = None
    if policy:
        policy_pass_presets = policy.get("pass_presets") or policy.get("passes")
    if policy_pass_presets:
        context_map.setdefault("policy_pass_presets", policy_pass_presets)
        for idx, pass_cfg in enumerate(policy_pass_presets):
            if not isinstance(pass_cfg, Mapping):
                continue
            try:
                preset_result = apply_tuning_preset(model, dict(pass_cfg))
                model = preset_result
                applied_pass_presets.append(dict(pass_cfg))
            except Exception:
                logger.exception("Failed to apply policy pass preset #%s", idx)
    if applied_pass_presets:
        context_map["applied_pass_presets"] = applied_pass_presets

    try:
        telemetry = collect_telemetry(model, input_tensor)
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("Telemetry collection failed")
        return model
    context_map["telemetry_event_count"] = len(telemetry)

    try:
        ir_nodes = extract_ir(model, telemetry)
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("IR extraction failed")
        return model
    context_map["ir_node_count"] = len(ir_nodes)

    try:
        llm_model_name = None
        if policy:
            llm_model_name = policy.get("llm_model")
        suggestion = request_kernel_suggestions(telemetry, ir_nodes, client=client, model_name=llm_model_name or "gpt-5-codex")
        if suggestion:
            logger.info("LLM suggestion: %s", suggestion)
            context_map["llm_suggestion_raw"] = suggestion
            try:
                context_map["llm_suggestion"] = json.loads(suggestion)
            except Exception:
                pass
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("LLM call failed")
        return model

    if enable_rl:
        # Optional: feature-flag the Codex-guided RL so tests and
        # environments without network/deps are not affected by default.
        if os.getenv("AGNITRA_ENABLE_LLM_RL") == "1":
            try:
                preset_candidate = run_llm_guided_rl(telemetry, ir_nodes, client=client)
                if preset_candidate:
                    applied_preset = dict(preset_candidate)
                    model = apply_tuning_preset(model, preset_candidate)
            except Exception:  # pragma: no cover - best effort
                logger.exception("LLM-guided RL failed")
            # Optionally skip PPO-based RL entirely when using LLM
            if os.getenv("AGNITRA_ONLY_LLM") == "1":
                context_map["applied_preset"] = applied_preset
                return model
        try:
            run_rl_tuning(telemetry, ir_nodes)
        except Exception:  # pragma: no cover - exercised via tests
            logger.exception("RL tuning failed")
            return model

    context_map["applied_preset"] = applied_preset
    setattr(model, "_agnitra_last_optimization", context_map)
    return model
