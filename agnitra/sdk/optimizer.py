import json
import logging
from typing import Any, Dict, List, Optional
import os

import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.profiler import ProfilerActivity, profile, record_function

from .deps import require_openai, require_sb3
from agnitra.core.optimizer import (
    PPOKernelOptimizer,
    PPOKernelOptimizerConfig,
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


def extract_ir(model: nn.Module, telemetry: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract a simple IR using torch.fx and attach telemetry."""
    traced = symbolic_trace(model)
    ir_nodes: List[Dict[str, Any]] = []
    for node in traced.graph.nodes:
        matched = next((t for t in telemetry if node.target and node.target in str(t["name"])), None)
        ir_nodes.append(
            {
                "op": node.op,
                "target": str(node.target),
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
    model_name: str = "codex-latest",
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
    response = client.responses.create(
        model=_model, input=[system_message, user_message], max_output_tokens=1024, store=False
    )
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


def optimize_model(
    model: nn.Module,
    input_tensor: torch.Tensor,
    client: Optional[Any] = None,
    enable_rl: bool = True,
) -> nn.Module:
    """Run the optimization pipeline with graceful fallbacks.

    On any stage failure, the exception is logged and the baseline model is returned
    untouched.
    """
    try:
        telemetry = collect_telemetry(model, input_tensor)
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("Telemetry collection failed")
        return model

    try:
        ir_nodes = extract_ir(model, telemetry)
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("IR extraction failed")
        return model

    try:
        suggestion = request_kernel_suggestions(telemetry, ir_nodes, client=client)
        if suggestion:
            logger.info("LLM suggestion: %s", suggestion)
    except Exception:  # pragma: no cover - exercised via tests
        logger.exception("LLM call failed")
        return model

    if enable_rl:
        # Optional: feature-flag the Codex-guided RL so tests and
        # environments without network/deps are not affected by default.
        if os.getenv("AGNITRA_ENABLE_LLM_RL") == "1":
            try:
                preset = run_llm_guided_rl(telemetry, ir_nodes, client=client)
                if preset:
                    model = apply_tuning_preset(model, preset)
            except Exception:  # pragma: no cover - best effort
                logger.exception("LLM-guided RL failed")
            # Optionally skip PPO-based RL entirely when using LLM
            if os.getenv("AGNITRA_ONLY_LLM") == "1":
                return model
        try:
            run_rl_tuning(telemetry, ir_nodes)
        except Exception:  # pragma: no cover - exercised via tests
            logger.exception("RL tuning failed")
            return model

    return model
