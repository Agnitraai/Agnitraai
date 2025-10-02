"""Codex-guided RL agent.

This agent uses an LLM (e.g., OpenAI Codex) to propose kernel/runtime tuning
presets from telemetry + IR, then optionally evaluates them with a tiny RL
loop. It is designed to be lightweight and optional: if the OpenAI client or
SB3 are not available, it degrades gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _safe_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        return None


@dataclass
class CodexGuidedAgent:
    """LLM-in-the-loop advisor for RL tuning presets.

    Workflow:
    1) Request a tuning preset from an LLM given telemetry + IR graph
       (constrained JSON schema).
    2) Optionally run a tiny RL-based evaluator to pick the best among a small
       set of candidate presets.
    """

    model_name: str = "gpt-5-codex"

    def _require_openai(self):  # lazy import to avoid hard dependency
        from agnitra._sdk.deps import require_openai

        try:
            OpenAI = require_openai()
        except Exception as exc:  # pragma: no cover - optional
            logger.info("%s", exc)
            return None
        return OpenAI

    def propose_config(
        self,
        telemetry: List[Dict[str, Any]],
        ir_nodes: List[Dict[str, Any]],
        client: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ask an LLM for a single tuning preset. Returns a config dict or None.

        The expected JSON schema is:
          {"allow_tf32": bool, "flash_sdp": bool, "torch_compile": bool, "kv_cache_dtype": "fp16"|"fp8"|"bf16"}
        """

        if client is None:
            OpenAI = self._require_openai()
            if OpenAI is None:
                return None
            client = OpenAI()

        try:
            ir_json = json.dumps(ir_nodes)
        except TypeError:
            ir_min = [{"op": n.get("op"), "target": n.get("target")} for n in ir_nodes]
            ir_json = json.dumps(ir_min)

        system = {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "You are an expert GPU kernel/runtime tuner. Output ONLY a JSON object with keys: "
                        "allow_tf32 (bool), flash_sdp (bool), torch_compile (bool), kv_cache_dtype (one of fp16, fp8, bf16)."
                    ),
                }
            ],
        }
        user = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        "Given telemetry and IR, choose a preset that reduces latency on A100/H100 while preserving quality.\n"
                        f"Telemetry: {telemetry}\nIR: {ir_json}"
                    ),
                }
            ],
        }
        try:
            import os as _os
            _model = _os.getenv("AGNITRA_LLM_MODEL", self.model_name)
            resp = client.responses.create(model=_model, input=[system, user], store=False)
        except Exception as exc:  # pragma: no cover - network/availability
            logger.info("Codex request failed: %s", exc)
            return None

        text = ""
        try:
            for item in getattr(resp, "output", []) or []:
                for entry in getattr(item, "content", []) or []:
                    text += getattr(entry, "text", "") or ""
        except Exception:
            logger.info("Unexpected Codex response schema; skipping parse")
            return None

        candidate = _safe_json(text.strip()) or {}
        allow_tf32 = bool(candidate.get("allow_tf32", True))
        flash_sdp = bool(candidate.get("flash_sdp", True))
        torch_compile = bool(candidate.get("torch_compile", False))
        kv_cache_dtype = str(candidate.get("kv_cache_dtype", "fp16")).lower()
        if kv_cache_dtype not in {"fp16", "bf16", "fp8"}:
            kv_cache_dtype = "fp16"
        return {
            "allow_tf32": allow_tf32,
            "flash_sdp": flash_sdp,
            "torch_compile": torch_compile,
            "kv_cache_dtype": kv_cache_dtype,
        }

    def evaluate_with_sb3(
        self, telemetry: List[Dict[str, Any]], candidates: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Pick a candidate using a tiny SB3 evaluator. Returns the chosen config or None."""
        try:
            from stable_baselines3 import PPO  # type: ignore
            import gymnasium as gym  # type: ignore
            import numpy as np  # type: ignore
        except Exception as exc:  # pragma: no cover - optional
            logger.info("SB3 not available: %s", exc)
            return candidates[0] if candidates else None

        total_cuda_ms = 0.0
        for evt in telemetry:
            total_cuda_ms += float(evt.get("cuda_time_ms", 0.0))
        total_cuda_ms = float(total_cuda_ms)

        class _Env(gym.Env):  # type: ignore[misc]
            metadata = {"render_modes": []}

            def __init__(self) -> None:
                super().__init__()
                self.action_space = gym.spaces.Discrete(len(candidates) or 1)
                self.observation_space = gym.spaces.Box(
                    low=0.0, high=np.finfo(np.float32).max, shape=(1,), dtype=np.float32
                )
                self.state = np.array([total_cuda_ms], dtype=np.float32)
                self._best = (float("inf"), 0)

            def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
                super().reset(seed=seed)
                self.state[:] = total_cuda_ms
                return self.state, {}

            def step(self, action: int):  # type: ignore[override]
                # Simulate improvement factors per candidate
                factors = [0.75, 0.8, 0.85, 0.9][: len(candidates)] or [0.9]
                simulated = float(self.state[0]) * factors[action]
                reward = (self.state[0] - simulated)
                done = True
                if simulated < self._best[0]:
                    self._best = (simulated, int(action))
                return self.state, float(reward), done, False, {}

            def best_action(self) -> int:
                return int(self._best[1])

        env = _Env()
        try:
            agent = PPO("MlpPolicy", env, verbose=0, n_steps=2, batch_size=4, n_epochs=1)
            agent.learn(total_timesteps=5)
            a = env.best_action()
            return candidates[a]
        except Exception as exc:  # pragma: no cover - best effort
            logger.info("SB3 eval failed: %s", exc)
            return candidates[0] if candidates else None
        finally:
            env.close()
