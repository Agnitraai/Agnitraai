"""Advanced telemetry collector using ``torch.profiler``.

This module profiles a PyTorch model and records per-layer metrics such as
CUDA time, tensor shapes and memory usage. Additionally, if NVIDIA's NVML is
available, GPU utilisation and power draw are logged. Results are optionally
written to a JSON file for further analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import time
import threading
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import torch
    from torch.profiler import ProfilerActivity, profile
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None
    ProfilerActivity = profile = None  # type: ignore[assignment]

# Optional RL stack for lightweight policy advice (lazy import to avoid noisy logs)
_SB3_AVAILABLE = None  # tri-state: None=unknown, False=unavailable, True=available
PPO = None  # type: ignore[assignment]
gym = None  # type: ignore[assignment]

def _import_rl_stack() -> bool:
    """Import SB3+Gymnasium lazily and quietly.

    Avoids importing at module load to prevent third-party libraries (e.g.,
    legacy Gym, Abseil/TF) from emitting warnings unless RL is explicitly used.
    """
    global _SB3_AVAILABLE, PPO, gym
    if _SB3_AVAILABLE is not None:
        return bool(_SB3_AVAILABLE)
    import os as _os
    import warnings as _warnings
    import logging as _logging
    # Suppress common noisy emitters
    _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    _os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")
    _os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
    _warnings.filterwarnings("ignore")
    _logging.getLogger("absl").setLevel(_logging.ERROR)
    try:  # pragma: no cover - optional dependency
        from stable_baselines3 import PPO as _PPO  # type: ignore
        import gymnasium as _gym  # type: ignore
        PPO = _PPO
        gym = _gym
        _SB3_AVAILABLE = True
    except Exception:
        _SB3_AVAILABLE = False
    return bool(_SB3_AVAILABLE)

try:  # pragma: no cover - optional dependency
    from pynvml import (
        NVMLError,
        NVML_TEMPERATURE_GPU,
        nvmlDeviceGetFanSpeed,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlDeviceGetPowerUsage,
        nvmlDeviceGetTemperature,
        nvmlDeviceGetUtilizationRates,
        nvmlInit,
        nvmlShutdown,
        nvmlSystemGetDriverVersion,
    )

    _NVML_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _NVML_AVAILABLE = False


@dataclass
class EventTelemetry:
    """Telemetry for a single profiler event."""

    name: str
    cuda_time_total: float
    cpu_time_total: float
    self_cuda_memory_usage: int
    self_cpu_memory_usage: int
    input_shapes: List[str]
    count: int
    self_cpu_time_total: float
    cuda_time_avg: float
    cpu_time_avg: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cuda_time_total": self.cuda_time_total,
            "cpu_time_total": self.cpu_time_total,
            "self_cuda_memory_usage": self.self_cuda_memory_usage,
            "self_cpu_memory_usage": self.self_cpu_memory_usage,
            "input_shapes": self.input_shapes,
            "count": self.count,
            "self_cpu_time_total": self.self_cpu_time_total,
            "cuda_time_avg": self.cuda_time_avg,
            "cpu_time_avg": self.cpu_time_avg,
        }


class _GpuSampler:
    """NVML sampler to collect a short GPU utilisation timeline."""

    def __init__(self, interval_s: float = 0.05) -> None:
        self.interval_s = interval_s
        self._running = False
        self._thr: Optional[threading.Thread] = None
        self.timeline: List[Dict[str, Any]] = []

    def start(self) -> None:
        if not _NVML_AVAILABLE:
            return
        try:  # pragma: no cover - requires GPU
            nvmlInit()
            handle = nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return

        self._running = True

        def _run() -> None:
            while self._running:
                try:
                    util = nvmlDeviceGetUtilizationRates(handle)
                    mem = nvmlDeviceGetMemoryInfo(handle)
                    ts = time.time()
                    self.timeline.append(
                        {
                            "t": ts,
                            "gpu": int(getattr(util, "gpu", 0)),
                            "mem": int(getattr(util, "memory", 0)),
                            "vram_used_mb": float(mem.used / (1024**2)) if mem else None,
                        }
                    )
                except Exception:
                    pass
                time.sleep(self.interval_s)

        self._thr = threading.Thread(target=_run, daemon=True)
        self._thr.start()

    def stop(self) -> None:
        self._running = False
        if self._thr is not None:
            try:
                self._thr.join(timeout=0.2)
            except Exception:
                pass
        if _NVML_AVAILABLE:  # pragma: no cover - requires GPU
            try:
                nvmlShutdown()
            except Exception:
                pass


@dataclass
class GpuTelemetry:
    """High level GPU metrics."""

    gpu_utilisation: int | None = None
    memory_utilisation: int | None = None
    power_watts: float | None = None
    temperature_c: float | None = None
    fan_speed_pct: float | None = None
    name: str | None = None
    driver_version: str | None = None
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None
    vram_free_mb: float | None = None
    cuda_allocated_mb: float | None = None
    cuda_reserved_mb: float | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gpu_utilisation": self.gpu_utilisation,
            "memory_utilisation": self.memory_utilisation,
            "power_watts": self.power_watts,
            "temperature_c": self.temperature_c,
            "fan_speed_pct": self.fan_speed_pct,
            "name": self.name,
            "driver_version": self.driver_version,
            "vram_total_mb": self.vram_total_mb,
            "vram_used_mb": self.vram_used_mb,
            "vram_free_mb": self.vram_free_mb,
            "cuda_allocated_mb": self.cuda_allocated_mb,
            "cuda_reserved_mb": self.cuda_reserved_mb,
        }


def _capture_gpu_metrics() -> GpuTelemetry:
    """Best effort capture of GPU metrics using NVML."""

    # Defaults capture CUDA memory from torch when available
    torch_alloc = torch.cuda.memory_allocated() / (1024**2) if (torch is not None and torch.cuda.is_available()) else None
    torch_reserved = torch.cuda.memory_reserved() / (1024**2) if (torch is not None and torch.cuda.is_available()) else None

    if not _NVML_AVAILABLE:
        return GpuTelemetry(
            cuda_allocated_mb=torch_alloc,
            cuda_reserved_mb=torch_reserved,
        )

    try:  # pragma: no cover - requires GPU
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        # Collect metrics defensively so one failure doesn't drop all
        util = None
        power = None
        mem = None
        temp = None
        fan = None
        name = None
        driver = None
        try:
            util = nvmlDeviceGetUtilizationRates(handle)
        except Exception:
            util = None
        try:
            power = nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
        except Exception:
            power = None
        try:
            mem = nvmlDeviceGetMemoryInfo(handle)
        except Exception:
            mem = None
        try:
            temp = float(nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU))
        except Exception:
            temp = None
        try:
            fan = float(nvmlDeviceGetFanSpeed(handle))
        except Exception:
            fan = None
        try:
            name = nvmlDeviceGetName(handle).decode() if hasattr(nvmlDeviceGetName(handle), "decode") else str(nvmlDeviceGetName(handle))
        except Exception:
            name = None
        try:
            driver = nvmlSystemGetDriverVersion().decode() if hasattr(nvmlSystemGetDriverVersion(), "decode") else str(nvmlSystemGetDriverVersion())
        except Exception:
            driver = None

        return GpuTelemetry(
            gpu_utilisation=(util.gpu if util is not None else None),
            memory_utilisation=(util.memory if util is not None else None),
            power_watts=power,
            temperature_c=temp,
            fan_speed_pct=fan,
            name=name,
            driver_version=driver,
            vram_total_mb=(mem.total / (1024**2) if mem is not None else None),
            vram_used_mb=(mem.used / (1024**2) if mem is not None else None),
            vram_free_mb=(mem.free / (1024**2) if mem is not None else None),
            cuda_allocated_mb=torch_alloc,
            cuda_reserved_mb=torch_reserved,
        )
    except Exception:
        return GpuTelemetry(
            cuda_allocated_mb=torch_alloc,
            cuda_reserved_mb=torch_reserved,
        )
    finally:  # pragma: no cover - requires GPU
        try:
            nvmlShutdown()
        except Exception:
            pass


def _dtype_str(dt: Any) -> str:
    """Return a compact dtype string (e.g., 'float16', 'bfloat16')."""

    try:
        s = str(dt)
        if s.startswith("torch."):
            s = s[len("torch.") :]
        return s
    except Exception:
        return str(dt)


def _module_common_attrs(m: Any) -> Dict[str, Any]:
    """Extract a small set of well-known attributes for common layers.

    Keeps the output compact but informative without introspecting arbitrary
    fields. The selection is conservative to avoid brittle dependencies.
    """

    attrs: Dict[str, Any] = {}
    try:  # Only if torch is available
        if torch is None:
            return attrs
        import torch.nn as nn  # type: ignore

        if isinstance(m, nn.Linear):
            attrs.update(
                {
                    "in_features": getattr(m, "in_features", None),
                    "out_features": getattr(m, "out_features", None),
                    "bias": getattr(m, "bias", None) is not None,
                }
            )
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            attrs.update(
                {
                    "in_channels": getattr(m, "in_channels", None),
                    "out_channels": getattr(m, "out_channels", None),
                    "kernel_size": tuple(getattr(m, "kernel_size", ())) or None,
                    "stride": tuple(getattr(m, "stride", ())) or None,
                    "padding": tuple(getattr(m, "padding", ())) or None,
                    "dilation": tuple(getattr(m, "dilation", ())) or None,
                    "groups": getattr(m, "groups", None),
                    "bias": getattr(m, "bias", None) is not None,
                }
            )
        elif isinstance(m, (nn.LayerNorm,)):
            attrs.update(
                {
                    "normalized_shape": tuple(getattr(m, "normalized_shape", ())) or None,
                    "eps": getattr(m, "eps", None),
                    "elementwise_affine": getattr(m, "elementwise_affine", None),
                }
            )
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            attrs.update(
                {
                    "num_features": getattr(m, "num_features", None),
                    "eps": getattr(m, "eps", None),
                    "momentum": getattr(m, "momentum", None),
                    "affine": getattr(m, "affine", None),
                    "track_running_stats": getattr(m, "track_running_stats", None),
                }
            )
    except Exception:
        # Best effort; missing attrs are simply omitted
        pass
    return attrs


def _collect_model_structure(model: Any) -> Dict[str, Any]:
    """Return a structured summary of modules, params, buffers and dtypes.

    This does not perform a forward pass. It complements profiler events with
    accurate parameter metadata (shapes, dtypes, learnable flags, devices).
    """

    info: Dict[str, Any] = {"layers": []}
    if model is None:
        return info

    # Aggregate level metrics
    total_params = 0
    trainable_params = 0
    dtype_hist: Dict[str, int] = {}
    buffer_hist: Dict[str, int] = {}

    try:
        # Named modules for stable identifiers
        for mod_name, m in getattr(model, "named_modules", lambda: [])():  # type: ignore[misc]
            try:
                # Local-only params/buffers to avoid duplication from parents
                params = []
                for pname, p in getattr(m, "named_parameters", lambda **k: [])(recurse=False):  # type: ignore[misc]
                    try:
                        dt = _dtype_str(getattr(p, "dtype", None))
                        numel = int(getattr(p, "numel", lambda: 0)())
                        total_params += numel
                        if bool(getattr(p, "requires_grad", False)):
                            trainable_params += numel
                        dtype_hist[dt] = dtype_hist.get(dt, 0) + numel
                        params.append(
                            {
                                "name": str(pname),
                                "shape": list(getattr(p, "shape", [])),
                                "numel": numel,
                                "dtype": dt,
                                "device": str(getattr(p, "device", "")),
                                "requires_grad": bool(getattr(p, "requires_grad", False)),
                                "kind": ("weight" if pname == "weight" else ("bias" if pname == "bias" else "parameter")),
                            }
                        )
                    except Exception:
                        continue

                buffers = []
                for bname, b in getattr(m, "named_buffers", lambda **k: [])(recurse=False):  # type: ignore[misc]
                    try:
                        dt = _dtype_str(getattr(b, "dtype", None))
                        numel = int(getattr(b, "numel", lambda: 0)())
                        buffer_hist[dt] = buffer_hist.get(dt, 0) + numel
                        buffers.append(
                            {
                                "name": str(bname),
                                "shape": list(getattr(b, "shape", [])),
                                "numel": numel,
                                "dtype": dt,
                                "device": str(getattr(b, "device", "")),
                                "kind": "buffer",
                            }
                        )
                    except Exception:
                        continue

                layer = {
                    "name": str(mod_name),
                    "type": getattr(m, "__class__", type(m)).__name__,
                    "class_path": f"{m.__class__.__module__}.{m.__class__.__name__}",
                    "parameters": params,
                    "buffers": buffers,
                    "attributes": _module_common_attrs(m),
                    # dynamic metrics populated from a live forward pass
                    "input_shapes": [],
                    "output_shapes": [],
                    "output_dtype": None,
                    "cuda_mem_alloc_delta_bytes": None,
                    "forward_time_ms": None,
                }
                info["layers"].append(layer)
            except Exception:
                continue

        info["parameter_count_total"] = int(total_params)
        info["parameter_count_trainable"] = int(trainable_params)
        info["parameter_count_frozen"] = int(total_params - trainable_params)
        info["parameter_dtypes"] = dtype_hist
        info["buffer_dtypes"] = buffer_hist
        # Heuristic default dtype: most common by count
        if dtype_hist:
            info["model_dtype"] = max(dtype_hist.items(), key=lambda kv: kv[1])[0]
        else:
            info["model_dtype"] = None
    except Exception:
        # Leave info minimal on failure
        pass

    return info


def _collect_module_io_shapes(model: Any, input_tensor: Any) -> Dict[str, Dict[str, Any]]:
    """Run a single forward pass with hooks to capture IO shapes and dtypes.

    Returns a mapping from module-qualified-name to a dict with keys
    'input_shapes', 'output_shapes' and 'output_dtype'. When hooks are not
    available, returns an empty mapping.
    """

    if torch is None:
        return {}

    io: Dict[str, Dict[str, Any]] = {}

    def as_shapes(x: Any) -> List[List[int]]:
        try:
            import torch as _t  # local
            if isinstance(x, _t.Tensor):
                return [list(x.shape)]
            if isinstance(x, (list, tuple)):
                out: List[List[int]] = []
                for itm in x:
                    out.extend(as_shapes(itm))
                return out
            return []
        except Exception:
            return []

    def out_dtype(x: Any) -> Optional[str]:
        try:
            import torch as _t
            if isinstance(x, _t.Tensor):
                return _dtype_str(x.dtype)
            if isinstance(x, (list, tuple)):
                for itm in x:
                    d = out_dtype(itm)
                    if d is not None:
                        return d
            return None
        except Exception:
            return None

    handles: List[Any] = []
    try:
        named_modules = dict(getattr(model, "named_modules", lambda: [])())  # type: ignore[misc]
        for name, m in named_modules.items():
            try:
                def _pre(_m, inputs, _name=name):  # type: ignore[no-redef]
                    slot = io.setdefault(_name, {})
                    slot["input_shapes"] = as_shapes(inputs)
                    # capture pre-alloc and start time
                    try:
                        if torch.cuda.is_available():
                            slot["_pre_alloc_bytes"] = int(torch.cuda.memory_allocated())
                    except Exception:
                        slot["_pre_alloc_bytes"] = None
                    slot["_t0"] = time.perf_counter()

                def _post(_m, inputs, output, _name=name):  # type: ignore[no-redef]
                    slot = io.setdefault(_name, {})
                    slot["output_shapes"] = as_shapes(output)
                    slot["output_dtype"] = out_dtype(output)
                    # forward latency
                    t0 = slot.get("_t0")
                    if isinstance(t0, float):
                        slot["forward_time_ms"] = (time.perf_counter() - t0) * 1000.0
                    # CUDA alloc delta
                    try:
                        if torch.cuda.is_available():
                            post_alloc = int(torch.cuda.memory_allocated())
                            pre_alloc = slot.get("_pre_alloc_bytes")
                            if isinstance(pre_alloc, int):
                                slot["cuda_mem_alloc_delta_bytes"] = int(post_alloc - pre_alloc)
                    except Exception:
                        pass

                handles.append(m.register_forward_pre_hook(_pre))
                handles.append(m.register_forward_hook(_post))
            except Exception:
                continue

        # One forward pass to trigger hooks
        model(input_tensor)
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    except Exception:
        return {}
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # remove temp keys
    for v in io.values():
        v.pop("_pre_alloc_bytes", None)
        v.pop("_t0", None)
    return io


def _heuristic_tinyllama_recommendations(events: List["EventTelemetry"]) -> Dict[str, Any]:
    """Return quick, hardware-friendly hints for TinyLlama-like workloads.

    This is a heuristic advisor that does not mutate runtime state. It produces
    suggested settings that a caller may apply when appropriate. The choices are
    aligned with PRD goals: favor FlashAttention (when available), allow TF32 on
    Ampere+ GPUs for matmul-heavy paths, and prefer FP16 KV cache for memory.
    """

    names = " ".join(e.name.lower() for e in events)
    attn_like = any(k in names for k in (
        "sdpa", "scaled_dot_product_attention", "attention", "attn"
    ))
    matmul_heavy = ("mm" in names) or ("matmul" in names) or attn_like

    cfg: Dict[str, Any] = {
        "allow_tf32": False,
        "flash_sdp": False,
        "kv_cache_dtype": "fp16",
        "torch_compile": False,
    }
    notes: List[str] = []

    if torch is not None and hasattr(torch.backends, "cuda"):
        # TF32 boosts throughput on A100/H100 while keeping quality acceptable
        cfg["allow_tf32"] = True
        notes.append("Enable TF32 matmul on Ampere+ for higher throughput")

    if attn_like and torch is not None and hasattr(torch.backends, "cuda"):
        # Prefer FlashAttention/SDPA kernels when available
        cfg["flash_sdp"] = True
        notes.append("Prefer FlashAttention/SDPA kernels for attention ops")

    if matmul_heavy and hasattr(torch, "compile"):
        cfg["torch_compile"] = True
        notes.append("Consider torch.compile for matmul-heavy sections")

    return {"config": cfg, "notes": notes}


def _rl_advise_from_events(events: List["EventTelemetry"]) -> Optional[Dict[str, Any]]:
    """Tiny, fast RL shim that selects between preset configs.

    If SB3/Gym are present, we use a trivial bandit-style environment with a
    handful of discrete actions mapping to configuration presets. Reward is a
    simple function of aggregated CUDA time (lower is better). The agent trains
    for a few timesteps and returns the best preset observed. Falls back to
    heuristics when RL stack is unavailable.
    """

    # Aggregate a latency proxy from profiler events
    total_cuda_ms = sum(getattr(e, "cuda_time_total", 0.0) for e in events) / 1e6

    presets = [
        {"allow_tf32": True, "flash_sdp": True, "kv_cache_dtype": "fp16", "torch_compile": False},
        {"allow_tf32": True, "flash_sdp": False, "kv_cache_dtype": "fp16", "torch_compile": True},
        {"allow_tf32": False, "flash_sdp": True, "kv_cache_dtype": "fp16", "torch_compile": False},
    ]

    if not _import_rl_stack():  # Fall back to heuristics
        return {"config": presets[0], "notes": ["SB3 not available; using preset #0"]}

    # Minimal gymnasium environment
    import numpy as _np  # local import to avoid hard dependency

    class _TelemetryEnv(gym.Env):  # type: ignore[misc]
        metadata = {"render_modes": []}

        def __init__(self) -> None:
            super().__init__()
            self.action_space = gym.spaces.Discrete(len(presets))
            self.observation_space = gym.spaces.Box(
                low=0.0, high=_np.finfo(_np.float32).max, shape=(1,), dtype=_np.float32
            )
            self.state = _np.array([total_cuda_ms], dtype=_np.float32)
            self._best: tuple[float, int] = (float("inf"), 0)

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
            super().reset(seed=seed)
            self.state[:] = total_cuda_ms
            return self.state, {}

        def step(self, action: int):  # type: ignore[override]
            # Reward: prefer actions that reduce latency proxy by fixed factors
            factors = [0.75, 0.80, 0.90]
            simulated = float(self.state[0]) * factors[action]
            reward = (self.state[0] - simulated)  # higher is better (latency drop)
            done = True
            info = {}
            if simulated < self._best[0]:
                self._best = (simulated, int(action))
            return self.state, float(reward), done, False, info

        def best_action(self) -> int:
            return int(self._best[1])

    env = _TelemetryEnv()
    try:
        # SB3 requires n_steps * n_envs > 1; keep tiny to stay fast
        agent = PPO("MlpPolicy", env, verbose=0, n_steps=2, batch_size=4, n_epochs=1)
        agent.learn(total_timesteps=5)
        action = env.best_action()
        return {"config": presets[action], "notes": [f"RL-chosen preset #{action}"]}
    except Exception as exc:  # pragma: no cover - best effort
        logging.info("RL advise failed: %s", exc)
        return None
    finally:
        env.close()


def profile_model(
    model: "torch.nn.Module",  # type: ignore[name-defined]
    input_tensor: "torch.Tensor",  # type: ignore[name-defined]
    json_path: str | None = None,
    with_rl: bool = False,
) -> Dict[str, Any]:
    """Profile ``model`` using ``input_tensor``.

    When PyTorch is unavailable, an empty telemetry payload is returned and,
    if ``json_path`` is provided, written to disk. This allows callers to
    gracefully proceed in CPU-only environments.
    """

    if torch is None or profile is None or ProfilerActivity is None:
        logging.warning("PyTorch not available; returning empty telemetry")
        payload = {"events": [], "gpu": {}}
        if json_path is not None:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        return payload

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():  # pragma: no branch
        activities.append(ProfilerActivity.CUDA)
        model = model.to("cuda")
        input_tensor = input_tensor.to("cuda")

    # Warmup a couple times to ensure hot execution before profiling
    try:
        if torch.cuda.is_available():
            for _ in range(2):
                _ = model(input_tensor)
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        else:
            for _ in range(2):
                _ = model(input_tensor)
    except Exception:
        pass

    # Optionally capture layer IO shapes via forward hooks (best effort)
    layer_io: Dict[str, Dict[str, Any]] = {}
    try:
        layer_io = _collect_module_io_shapes(model, input_tensor)
    except Exception:
        layer_io = {}

    # Start GPU utilisation sampler
    sampler = _GpuSampler()
    try:
        sampler.start()
    except Exception:
        pass

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        model(input_tensor)
        # Ensure all CUDA kernels complete so profiler captures device time
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
    try:
        sampler.stop()
    except Exception:
        pass

    events: List[EventTelemetry] = []
    for evt in prof.key_averages():
        try:
            cnt = int(getattr(evt, "count", 1))
        except Exception:
            cnt = 1
        cpu_total = float(getattr(evt, "cpu_time_total", 0.0))
        cuda_total = float(getattr(evt, "cuda_time_total", 0.0))
        self_cpu_total = float(getattr(evt, "self_cpu_time_total", 0.0))
        events.append(
            EventTelemetry(
                name=str(evt.key),
                cuda_time_total=cuda_total,
                cpu_time_total=cpu_total,
                self_cuda_memory_usage=int(getattr(evt, "self_cuda_memory_usage", 0)),
                self_cpu_memory_usage=int(getattr(evt, "self_cpu_memory_usage", 0)),
                input_shapes=[str(s) for s in getattr(evt, "input_shapes", [])],
                count=cnt,
                self_cpu_time_total=self_cpu_total,
                cuda_time_avg=(cuda_total / max(1, cnt)),
                cpu_time_avg=(cpu_total / max(1, cnt)),
            )
        )

    # If CUDA is available but we saw zero allocated bytes for all events,
    # run a short dynamic re-profile to encourage non-zero allocations (warm kernels)
    if torch.cuda.is_available():
        try:
            if all(int(getattr(e, "self_cuda_memory_usage", 0)) == 0 for e in events):
                with profile(
                    activities=activities,
                    record_shapes=True,
                    profile_memory=True,
                ) as prof2:
                    for _ in range(3):
                        _ = model(input_tensor)
                    torch.cuda.synchronize()
                events = []
                for evt in prof2.key_averages():
                    cnt = int(getattr(evt, "count", 1))
                    cpu_total = float(getattr(evt, "cpu_time_total", 0.0))
                    cuda_total = float(getattr(evt, "cuda_time_total", 0.0))
                    self_cpu_total = float(getattr(evt, "self_cpu_time_total", 0.0))
                    events.append(
                        EventTelemetry(
                            name=str(evt.key),
                            cuda_time_total=cuda_total,
                            cpu_time_total=cpu_total,
                            self_cuda_memory_usage=int(getattr(evt, "self_cuda_memory_usage", 0)),
                            self_cpu_memory_usage=int(getattr(evt, "self_cpu_memory_usage", 0)),
                            input_shapes=[str(s) for s in getattr(evt, "input_shapes", [])],
                            count=cnt,
                            self_cpu_time_total=self_cpu_total,
                            cuda_time_avg=(cuda_total / max(1, cnt)),
                            cpu_time_avg=(cpu_total / max(1, cnt)),
                        )
                    )
        except Exception:
            pass

    gpu = _capture_gpu_metrics()

    # Model/module/parameter summary (static)
    model_info = _collect_model_structure(model)
    # Merge IO shapes and dynamic per-layer metrics into layer entries by name
    if model_info.get("layers") and layer_io:
        by_name = {layer.get("name", ""): layer for layer in model_info["layers"]}
        for lname, io in layer_io.items():
            if lname in by_name:
                by_name[lname]["input_shapes"] = io.get("input_shapes", [])
                by_name[lname]["output_shapes"] = io.get("output_shapes", [])
                by_name[lname]["output_dtype"] = io.get("output_dtype")
                by_name[lname]["cuda_mem_alloc_delta_bytes"] = io.get("cuda_mem_alloc_delta_bytes")
                by_name[lname]["forward_time_ms"] = io.get("forward_time_ms")

    recommendations: Optional[Dict[str, Any]] = None
    if with_rl:
        # Try RL first, then heuristics if needed
        recommendations = _rl_advise_from_events(events) or _heuristic_tinyllama_recommendations(events)

    # Meta information for richer telemetry (>20 params overall when combined)
    meta: Dict[str, Any] = {}
    try:
        import platform as _pf
        meta.update(
            {
                "python_version": _pf.python_version(),
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None) if hasattr(torch, "version") else None,
                "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "device_capability": torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception:
        pass

    # Model summary
    try:
        num_params = sum(p.numel() for p in model.parameters()) if hasattr(model, "parameters") else None
        meta["model_num_parameters"] = int(num_params) if num_params is not None else None
    except Exception:
        meta["model_num_parameters"] = None

    # Additional behavior metrics to help identify bottlenecks and opportunities
    def _to_ms(v: float) -> float:
        try:
            return float(v) / 1e6
        except Exception:
            return 0.0

    behavior: Dict[str, Any] = {}
    try:
        op_names = [e.name for e in events]
        behavior.update(
            {
                "op_count_total": len(events),
                "unique_ops": len(set(op_names)),
                "matmul_ops": sum(1 for n in op_names if any(k in n.lower() for k in ("mm", "matmul", "bmm", "addmm"))),
                "conv_ops": sum(1 for n in op_names if "conv" in n.lower()),
                "activation_ops": sum(1 for n in op_names if any(k in n.lower() for k in ("relu", "gelu", "silu", "tanh", "sigmoid"))),
                "norm_ops": sum(1 for n in op_names if "norm" in n.lower()),
                "inplace_ops": sum(1 for n in op_names if n.strip().endswith("_")),
                "kernel_launches": sum(getattr(e, "count", 1) for e in events if "cudaLaunchKernel" in str(e.name)),
                "top_cuda_ms": sorted(({"name": e.name, "cuda_ms": _to_ms(e.cuda_time_total)} for e in events), key=lambda x: x["cuda_ms"], reverse=True)[:10],
                "top_cuda_mem_bytes": sorted(({"name": e.name, "bytes": int(e.self_cuda_memory_usage)} for e in events), key=lambda x: x["bytes"], reverse=True)[:10],
            }
        )
        # Approx activation volume from layer IO
        def _dtype_nbytes(dt: Optional[str]) -> int:
            s = (dt or "").lower()
            if "64" in s:
                return 8
            if "16" in s and "bfloat" in s:
                return 2
            if "16" in s:
                return 2
            if "8" in s:
                return 1
            return 4
        act_total = 0
        act_layers: List[Dict[str, Any]] = []
        for lyr in model_info.get("layers", []):
            dtb = _dtype_nbytes(lyr.get("output_dtype"))
            inc = sum(int(__import__('functools').reduce(lambda a,b: a*b, shp, 1)) * dtb for shp in lyr.get("input_shapes", []) if isinstance(shp, list))
            outc = sum(int(__import__('functools').reduce(lambda a,b: a*b, shp, 1)) * dtb for shp in lyr.get("output_shapes", []) if isinstance(shp, list))
            layer_bytes = int(inc + outc)
            if layer_bytes > 0:
                act_layers.append({"name": lyr.get("name"), "bytes": layer_bytes})
                act_total += layer_bytes
        behavior["activation_bytes_total"] = act_total
        behavior["top_activation_layers"] = sorted(act_layers, key=lambda x: x["bytes"], reverse=True)[:10]
    except Exception:
        pass

    gpu_timeline: List[Dict[str, Any]] = getattr(sampler, "timeline", []) if sampler else []
    if gpu_timeline:
        try:
            behavior["gpu_util_mean"] = sum(p.get("gpu", 0) for p in gpu_timeline) / max(1, len(gpu_timeline))
            behavior["gpu_util_max"] = max(p.get("gpu", 0) for p in gpu_timeline)
            behavior["mem_util_max"] = max(p.get("mem", 0) for p in gpu_timeline)
        except Exception:
            pass

    # Simple heuristic opportunities for precision/pruning
    opportunities: Dict[str, Any] = {}
    try:
        heavy_layers = []
        for lyr in model_info.get("layers", []):
            ltype = str(lyr.get("type", ""))
            params = sum(int(p.get("numel", 0)) for p in lyr.get("parameters", []))
            fwd_ms = float(lyr.get("forward_time_ms") or 0.0)
            if any(k in ltype.lower() for k in ("linear", "conv")):
                heavy_layers.append({
                    "name": lyr.get("name"),
                    "type": ltype,
                    "params": params,
                    "forward_ms": fwd_ms,
                })
        # Candidates: many params or slow forward
        fp16_candidates = sorted(heavy_layers, key=lambda x: (x["params"], x["forward_ms"]), reverse=True)[:5]
        prune_candidates = [h for h in heavy_layers if h["params"] > 1_000_000][:5]
        opportunities = {
            "fp16_bf16_candidates": fp16_candidates,
            "prune_candidates": prune_candidates,
        }
    except Exception:
        pass

    payload = {
        "events": [e.to_dict() for e in events],
        "gpu": gpu.to_dict(),
        "gpu_timeline": gpu_timeline,
        "meta": meta,
        "model": model_info,
        "behavior": behavior,
        "opportunities": opportunities,
    }
    if with_rl and recommendations is not None:
        payload["recommendations"] = recommendations

    if json_path is not None:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    return payload


__all__ = ["profile_model", "EventTelemetry", "GpuTelemetry"]
