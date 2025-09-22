"""Simulated reinforcement learning optimizer for kernel tuning.

This module exposes a gym-style environment that emulates GPU kernel feedback
(latency vs throughput) and a thin PPO-based training loop built on top of
stable-baselines3. It provides a lightweight way to experiment with automated
parameter search for tile size, loop unrolling, and fusion decisions without
requiring an actual accelerator runtime.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)

try:  # Optional dependency: numpy is only required when running the RL loop
    import numpy as _np
except Exception:  # pragma: no cover - optional dependency path
    _np = None  # type: ignore[misc]


@dataclass(frozen=True)
class KernelTuningSpace:
    """Discrete search space for kernel tuning primitives."""

    tile_sizes: Sequence[int] = (16, 32, 64, 128)
    unroll_factors: Sequence[int] = (1, 2, 4, 8)
    fuse_kernels: Sequence[bool] = (False, True)

    def to_action_dimensions(self) -> Tuple[int, int, int]:
        return (
            len(tuple(self.tile_sizes)) or 1,
            len(tuple(self.unroll_factors)) or 1,
            len(tuple(self.fuse_kernels)) or 1,
        )


@dataclass(frozen=True)
class KernelTelemetryStats:
    """Summary statistics derived from profiler telemetry."""

    baseline_latency_ms: float
    baseline_tokens_per_sec: float
    max_latency_ms: float
    total_latency_ms: float
    event_count: int


def _to_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_kernel_telemetry(
    telemetry: Sequence[Mapping[str, Any]] | None,
    *,
    default_latency_ms: float = 7.5,
    default_tokens_per_sec: float = 1100.0,
) -> KernelTelemetryStats:
    """Aggregate telemetry to seed the RL environment baselines."""

    events = tuple(telemetry or ())
    latency_candidates: List[float] = []
    token_candidates: List[float] = []
    for event in events:
        lat = None
        for key in ("cuda_time_ms", "latency_ms", "time_ms"):
            lat = _to_float(event.get(key))
            if lat is not None:
                break
        if lat is not None:
            latency_candidates.append(max(lat, 0.0))
        tokens = None
        for key in ("tokens_per_sec", "throughput_tps", "tokens/sec"):
            tokens = _to_float(event.get(key))
            if tokens is not None:
                break
        if tokens is not None and tokens > 0.0:
            token_candidates.append(tokens)

    if latency_candidates:
        baseline_latency = max(latency_candidates)
        max_latency = max(latency_candidates)
        total_latency = sum(latency_candidates)
    else:
        baseline_latency = default_latency_ms
        max_latency = default_latency_ms
        total_latency = default_latency_ms

    if token_candidates:
        baseline_tokens = max(token_candidates)
    else:
        fallback_tokens = 1000.0 / max(baseline_latency, 1e-3)
        baseline_tokens = max(default_tokens_per_sec, fallback_tokens * 32.0)

    return KernelTelemetryStats(
        baseline_latency_ms=float(baseline_latency),
        baseline_tokens_per_sec=float(baseline_tokens),
        max_latency_ms=float(max_latency),
        total_latency_ms=float(total_latency),
        event_count=len(events),
    )


@dataclass
class PPOKernelOptimizerConfig:
    """Configuration for the PPO-based kernel optimizer."""

    tuning_space: KernelTuningSpace = field(default_factory=KernelTuningSpace)
    max_episode_steps: int = 8
    reward_noise_std: float = 0.02
    baseline_latency_ms: float = 7.5
    baseline_tokens_per_sec: float = 1100.0
    randomize_baseline: bool = True
    min_latency_ms: float = 3.0
    max_latency_ms: float = 18.0
    min_tokens_per_sec: float = 600.0
    max_tokens_per_sec: float = 2400.0
    target_bonus_bounds: Tuple[float, float] = (0.08, 0.32)
    penalty_scale_bounds: Tuple[float, float] = (0.04, 0.12)
    total_timesteps: int = 6000
    learning_rate: float = 3e-4
    policy: str = "MlpPolicy"
    verbose: int = 0
    seed: Optional[int] = None
    telemetry_summary: Optional[KernelTelemetryStats] = None


@dataclass
class PPOKernelOptimizationResult:
    """Outcome of a PPO training run over the simulated environment."""

    tile_size: int
    unroll_factor: int
    fuse_kernels: bool
    tokens_per_sec: float
    latency_ms: float
    reward: float
    improvement_ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _EpisodeParameters:
    baseline_latency_ms: float
    baseline_tokens_per_sec: float
    target_signature: Tuple[int, int, int]
    target_bonus: float
    penalty_scale: float


def _sample_episode_parameters(
    config: PPOKernelOptimizerConfig,
    rng: "_np.random.Generator",
    tuning_space: KernelTuningSpace,
) -> _EpisodeParameters:
    summary = config.telemetry_summary
    if summary is not None:
        base_latency = summary.baseline_latency_ms
        base_tokens = summary.baseline_tokens_per_sec
        if config.randomize_baseline:
            base_latency *= float(rng.uniform(0.9, 1.1))
            base_tokens *= float(rng.uniform(0.9, 1.1))
    else:
        base_latency = float(
            rng.uniform(config.min_latency_ms, config.max_latency_ms)
        )
        base_tokens = float(
            rng.uniform(config.min_tokens_per_sec, config.max_tokens_per_sec)
        )
    base_latency = float(max(config.min_latency_ms, min(base_latency, config.max_latency_ms)))
    base_tokens = float(max(config.min_tokens_per_sec, min(base_tokens, config.max_tokens_per_sec)))
    dims = tuning_space.to_action_dimensions()
    target_signature = (
        int(rng.integers(dims[0])),
        int(rng.integers(dims[1])),
        int(rng.integers(dims[2])),
    )
    target_bonus = float(rng.uniform(*config.target_bonus_bounds))
    penalty_scale = float(rng.uniform(*config.penalty_scale_bounds))
    return _EpisodeParameters(
        baseline_latency_ms=base_latency,
        baseline_tokens_per_sec=base_tokens,
        target_signature=target_signature,
        target_bonus=target_bonus,
        penalty_scale=penalty_scale,
    )


def _evaluate_improvement(
    action: Tuple[int, int, int],
    tuning_space: KernelTuningSpace,
    params: _EpisodeParameters,
) -> float:
    tile_idx, unroll_idx, fusion_idx = action
    target_tile, target_unroll, target_fusion = params.target_signature
    tile_delta = abs(tile_idx - target_tile)
    unroll_delta = abs(unroll_idx - target_unroll)
    fusion_match = 1.0 if fusion_idx == target_fusion else -0.5
    penalty = params.penalty_scale * (tile_delta * 0.7 + unroll_delta * 0.5)
    penalty += 0.1 * fusion_match
    improvement = params.target_bonus - penalty
    tile_values = tuple(tuning_space.tile_sizes)
    if tile_idx < len(tile_values):
        tile = tile_values[tile_idx]
        if tile > 128:
            improvement -= 0.05 * math.log(tile / 128.0 + 1.0)
    return float(max(min(improvement, 0.4), -0.6))


def _simulate_metrics(
    improvement_ratio: float,
    params: _EpisodeParameters,
    config: PPOKernelOptimizerConfig,
    rng: "_np.random.Generator",
) -> Tuple[float, float, float, float]:
    noisy_ratio = improvement_ratio + float(
        rng.normal(0.0, config.reward_noise_std)
    )
    factor = max(0.1, 1.0 + noisy_ratio)
    tokens_per_sec = params.baseline_tokens_per_sec * factor
    latency_ms = params.baseline_latency_ms / factor
    reward = tokens_per_sec - params.baseline_tokens_per_sec
    return float(reward), float(tokens_per_sec), float(latency_ms), float(noisy_ratio)


class _KernelRuntimeEnv:
    """Gym-style environment simulating kernel runtime feedback."""

    metadata = {"render_modes": []}

    def __init__(self, config: PPOKernelOptimizerConfig) -> None:
        if _np is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "NumPy is required for the RL optimizer. Install with 'pip install agnitra[rl]'."
            )
        try:
            import gymnasium as gym  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "gymnasium is required for the RL optimizer. Install with 'pip install agnitra[rl]'."
            ) from exc

        self._gym = gym
        self.config = config
        self.tuning_space = config.tuning_space
        self._rng = _np.random.default_rng(config.seed)
        dims = self.tuning_space.to_action_dimensions()
        self.action_space = gym.spaces.MultiDiscrete(dims)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=_np.float32
        )
        self._step_count = 0
        self._episode_params = _sample_episode_parameters(self.config, self._rng, self.tuning_space)
        self._prev_action = _np.zeros(3, dtype=_np.int64)
        self._best_reward = -_np.inf
        self._best_action: Tuple[int, int, int] = (0, 0, 0)
        self._best_tokens_per_sec = self._episode_params.baseline_tokens_per_sec
        self._best_latency_ms = self._episode_params.baseline_latency_ms
        self._best_ratio = 0.0

    # -- gym API ---------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):  # type: ignore[override]
        if seed is not None:
            self._rng = _np.random.default_rng(seed)
        self._step_count = 0
        # Allow callers to override baselines via options while keeping defaults.
        if options and "telemetry_summary" in options:
            override = options["telemetry_summary"]
            if isinstance(override, KernelTelemetryStats):
                self.config.telemetry_summary = override
        self._episode_params = _sample_episode_parameters(self.config, self._rng, self.tuning_space)
        self._prev_action = _np.array(self.action_space.nvec // 2, dtype=_np.int64)
        self._best_reward = -_np.inf
        self._best_action = tuple(self._prev_action.tolist())
        self._best_tokens_per_sec = self._episode_params.baseline_tokens_per_sec
        self._best_latency_ms = self._episode_params.baseline_latency_ms
        self._best_ratio = 0.0
        obs = self._build_observation(0.0)
        info: Dict[str, Any] = {
            "baseline_latency_ms": self._episode_params.baseline_latency_ms,
            "baseline_tokens_per_sec": self._episode_params.baseline_tokens_per_sec,
            "target_signature": self._episode_params.target_signature,
        }
        return obs, info

    def step(self, action: Any):  # type: ignore[override]
        action_array = _np.asarray(action, dtype=_np.int64).reshape(-1)
        if action_array.size != 3:
            raise ValueError("Action must contain tile, unroll, and fusion indices")
        tile_idx, unroll_idx, fusion_idx = map(int, action_array.tolist())
        tile_idx = int(_np.clip(tile_idx, 0, self.action_space.nvec[0] - 1))
        unroll_idx = int(_np.clip(unroll_idx, 0, self.action_space.nvec[1] - 1))
        fusion_idx = int(_np.clip(fusion_idx, 0, self.action_space.nvec[2] - 1))
        self._step_count += 1

        improvement_ratio = _evaluate_improvement(
            (tile_idx, unroll_idx, fusion_idx), self.tuning_space, self._episode_params
        )
        reward, tokens_per_sec, latency_ms, noisy_ratio = _simulate_metrics(
            improvement_ratio, self._episode_params, self.config, self._rng
        )

        # Track best encountered configuration across episode rollouts.
        if reward > self._best_reward:
            self._best_reward = reward
            self._best_action = (tile_idx, unroll_idx, fusion_idx)
            self._best_tokens_per_sec = tokens_per_sec
            self._best_latency_ms = latency_ms
            self._best_ratio = noisy_ratio

        self._prev_action[:] = _np.array([tile_idx, unroll_idx, fusion_idx], dtype=_np.int64)
        terminated = self._step_count >= self.config.max_episode_steps
        truncated = False
        obs = self._build_observation(reward)
        info = {
            "tokens_per_sec": tokens_per_sec,
            "latency_ms": latency_ms,
            "improvement_ratio": noisy_ratio,
            "raw_improvement_ratio": improvement_ratio,
        }
        return obs, float(reward), bool(terminated), bool(truncated), info

    # -- helpers ---------------------------------------------------------
    def best_action_config(self) -> PPOKernelOptimizationResult:
        tile_values = tuple(self.tuning_space.tile_sizes)
        unroll_values = tuple(self.tuning_space.unroll_factors)
        fuse_values = tuple(self.tuning_space.fuse_kernels)
        tile_idx, unroll_idx, fusion_idx = self._best_action
        tile_size = tile_values[min(tile_idx, len(tile_values) - 1)]
        unroll_factor = unroll_values[min(unroll_idx, len(unroll_values) - 1)]
        fuse_flag = fuse_values[min(fusion_idx, len(fuse_values) - 1)]
        metadata = {
            "baseline_latency_ms": self._episode_params.baseline_latency_ms,
            "baseline_tokens_per_sec": self._episode_params.baseline_tokens_per_sec,
            "target_signature": self._episode_params.target_signature,
            "target_bonus": self._episode_params.target_bonus,
            "penalty_scale": self._episode_params.penalty_scale,
            "steps": self._step_count,
            "strategy": "ppo",
        }
        return PPOKernelOptimizationResult(
            tile_size=tile_size,
            unroll_factor=unroll_factor,
            fuse_kernels=bool(fuse_flag),
            tokens_per_sec=self._best_tokens_per_sec,
            latency_ms=self._best_latency_ms,
            reward=float(self._best_reward),
            improvement_ratio=float(self._best_ratio),
            metadata=metadata,
        )

    def _build_observation(self, reward: float) -> Any:
        max_tokens_delta = max(self._episode_params.baseline_tokens_per_sec, 1.0)
        normalized_reward = max(min(reward / max_tokens_delta + 0.5, 1.0), 0.0)
        step_fraction = self._step_count / max(1, self.config.max_episode_steps)
        tile_norm = self._prev_action[0] / max(1, self.action_space.nvec[0] - 1)
        unroll_norm = self._prev_action[1] / max(1, self.action_space.nvec[1] - 1)
        fusion_norm = self._prev_action[2]
        baseline_latency_norm = min(
            self._episode_params.baseline_latency_ms / (self.config.max_latency_ms + 1e-5), 1.0
        )
        best_norm = max(self._best_reward / max_tokens_delta + 0.5, 0.0)
        obs = _np.array(
            [
                float(normalized_reward),
                float(step_fraction),
                float(tile_norm),
                float(unroll_norm),
                float(fusion_norm),
                float(baseline_latency_norm + 0.05 * best_norm),
            ],
            dtype=_np.float32,
        )
        return obs

def _require_sb3_dependencies():
    if _np is None:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "NumPy is required for the RL optimizer. Install with 'pip install agnitra[rl]'."
        )
    try:
        from stable_baselines3 import PPO  # type: ignore
        from stable_baselines3.common.vec_env import DummyVecEnv  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "stable-baselines3 is required for the RL optimizer. Install with 'pip install agnitra[rl]'."
        ) from exc
    try:
        import gymnasium as gym  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "gymnasium is required for the RL optimizer. Install with 'pip install agnitra[rl]'."
        ) from exc
    return PPO, DummyVecEnv


class PPOKernelOptimizer:
    """Run a PPO agent over the simulated kernel runtime environment."""

    def __init__(self, config: Optional[PPOKernelOptimizerConfig] = None) -> None:
        self.config = config or PPOKernelOptimizerConfig()
        self._last_result: Optional[PPOKernelOptimizationResult] = None

    @property
    def last_result(self) -> Optional[PPOKernelOptimizationResult]:
        return self._last_result

    def train(self, total_timesteps: Optional[int] = None) -> PPOKernelOptimizationResult:
        try:
            PPO, DummyVecEnv = _require_sb3_dependencies()
        except RuntimeError as exc:
            LOGGER.info(
                "PPO dependencies unavailable (%s); falling back to heuristic search.",
                exc,
            )
            result = self._random_search(total_timesteps)
            self._last_result = result
            return result

        env_config = self.config

        def _make_env():
            return _KernelRuntimeEnv(env_config)

        vec_env = DummyVecEnv([_make_env])
        try:
            model = PPO(
                env=vec_env,
                policy=self.config.policy,
                learning_rate=self.config.learning_rate,
                verbose=self.config.verbose,
                seed=self.config.seed,
            )
            steps = int(total_timesteps or self.config.total_timesteps)
            if steps <= 0:
                raise ValueError("total_timesteps must be positive")
            model.learn(total_timesteps=steps)
            env = vec_env.envs[0]
            result = env.best_action_config()
            self._last_result = result
            return result
        finally:
            vec_env.close()


    def _random_search(self, total_timesteps: Optional[int]) -> PPOKernelOptimizationResult:
        if _np is None:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "NumPy is required for the RL optimizer fallback. Install with 'pip install agnitra[rl]'."
            )
        rng = _np.random.default_rng(self.config.seed)
        params = _sample_episode_parameters(self.config, rng, self.config.tuning_space)
        dims = self.config.tuning_space.to_action_dimensions()
        actions = _np.array(
            [(i, j, k) for i in range(dims[0]) for j in range(dims[1]) for k in range(dims[2])],
            dtype=_np.int64,
        )
        if actions.size == 0:
            actions = _np.zeros((1, 3), dtype=_np.int64)
        rng.shuffle(actions, axis=0)
        eval_limit = actions.shape[0]
        if total_timesteps is not None:
            eval_limit = int(max(1, min(eval_limit, total_timesteps)))
        else:
            eval_limit = int(max(1, min(eval_limit, self.config.max_episode_steps * 4)))

        best_reward = -_np.inf
        best_action = actions[0]
        best_tokens = params.baseline_tokens_per_sec
        best_latency = params.baseline_latency_ms
        best_ratio = 0.0
        for idx in actions[:eval_limit]:
            tile_idx, unroll_idx, fusion_idx = map(int, idx.tolist())
            improvement = _evaluate_improvement(
                (tile_idx, unroll_idx, fusion_idx), self.config.tuning_space, params
            )
            reward, tokens, latency, noisy_ratio = _simulate_metrics(
                improvement, params, self.config, rng
            )
            if reward > best_reward:
                best_reward = reward
                best_action = idx
                best_tokens = tokens
                best_latency = latency
                best_ratio = noisy_ratio

        tile_values = tuple(self.config.tuning_space.tile_sizes)
        unroll_values = tuple(self.config.tuning_space.unroll_factors)
        fuse_values = tuple(self.config.tuning_space.fuse_kernels)
        tile_idx, unroll_idx, fusion_idx = map(int, best_action.tolist())
        result = PPOKernelOptimizationResult(
            tile_size=tile_values[min(tile_idx, len(tile_values) - 1)] if tile_values else 1,
            unroll_factor=unroll_values[min(unroll_idx, len(unroll_values) - 1)] if unroll_values else 1,
            fuse_kernels=fuse_values[min(fusion_idx, len(fuse_values) - 1)] if fuse_values else False,
            tokens_per_sec=float(best_tokens),
            latency_ms=float(best_latency),
            reward=float(best_reward),
            improvement_ratio=float(best_ratio),
            metadata={
                "baseline_latency_ms": params.baseline_latency_ms,
                "baseline_tokens_per_sec": params.baseline_tokens_per_sec,
                "target_signature": params.target_signature,
                "target_bonus": params.target_bonus,
                "penalty_scale": params.penalty_scale,
                "steps": eval_limit,
                "strategy": "random-search",
                "dependency_fallback": True,
            },
        )
        return result


def run_dummy_training_loop(total_timesteps: int = 3000) -> PPOKernelOptimizationResult:
    """Convenience helper that runs a short PPO training session.

    This is intended for demos and unit tests; it does not require real hardware
    and finishes quickly while producing a plausible tuning configuration.
    """

    optimizer = PPOKernelOptimizer()
    result = optimizer.train(total_timesteps=total_timesteps)
    strategy = result.metadata.get("strategy", "ppo")
    LOGGER.info(
        "Dummy RL run (%s) selected tile=%s unroll=%s fuse=%s tokens/s=%.2f latency=%.2f (Δ=%.2f%%)",
        strategy,
        result.tile_size,
        result.unroll_factor,
        result.fuse_kernels,
        result.tokens_per_sec,
        result.latency_ms,
        result.improvement_ratio * 100.0,
    )
    if result.metadata.get("dependency_fallback"):
        LOGGER.info("Returned configuration from dependency fallback search mode.")
    return result


__all__ = [
    "KernelTelemetryStats",
    "KernelTuningSpace",
    "PPOKernelOptimizerConfig",
    "PPOKernelOptimizationResult",
    "PPOKernelOptimizer",
    "run_dummy_training_loop",
    "summarize_kernel_telemetry",
]
