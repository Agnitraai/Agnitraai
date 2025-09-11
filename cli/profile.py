"""Model profiling CLI with rich telemetry and progress.

This module provides a CLI-friendly entry point that:
- Loads a TorchScript model by path
- Notifies progress at each layer via forward hooks
- Profiles with torch.profiler (CUDA time, shapes, memory)
- Extends telemetry with NVML (GPU util, power, temp, VRAM)
- Writes JSON artifacts including a detailed result summary

Usage (via CLI):
    agnitra profile model.pt --input-shape 1,3,224,224 --output telemetry.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


GREEN = "\x1b[32m"
BLUE = "\x1b[34m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def _parse_shape(s: str) -> Sequence[int]:
    return tuple(int(x) for x in s.split(","))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class LayerLog:
    name: str
    start_t: float
    end_t: float | None = None
    elapsed_ms: float | None = None


def _register_progress_hooks(model: Any) -> Tuple[List[Any], Dict[int, LayerLog]]:
    """Register forward hooks that print progress and measure elapsed time.

    Returns a list of hook handles and a map from id(module) to LayerLog.
    """
    try:
        import torch  # noqa: F401
    except Exception:
        return [], {}

    handles: List[Any] = []
    logs: Dict[int, LayerLog] = {}

    # Map module id to fully qualified name for clarity
    name_by_id: Dict[int, str] = {}
    try:
        name_by_id = {id(m): n for n, m in model.named_modules()}
    except Exception:
        name_by_id = {}

    def _pre(module, inputs):  # type: ignore[no-untyped-def]
        mid = id(module)
        qname = name_by_id.get(mid) or getattr(module, "__class__", type(module)).__name__
        logs[mid] = LayerLog(name=qname, start_t=time.perf_counter())
        print(f"{BLUE}›{RESET} {qname} ...", flush=True)

    def _post(module, inputs, output):  # type: ignore[no-untyped-def]
        mid = id(module)
        log = logs.get(mid)
        if log is None:
            return
        log.end_t = time.perf_counter()
        log.elapsed_ms = (log.end_t - log.start_t) * 1000.0
        print(f"{GREEN}✔{RESET} {log.name} ({log.elapsed_ms:.3f} ms)", flush=True)

    try:
        for _, m in model.named_modules():
            try:
                handles.append(m.register_forward_pre_hook(_pre))
                handles.append(m.register_forward_hook(_post))
            except Exception:
                continue
    except Exception:
        return [], {}

    return handles, logs


def _summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    def to_ms_ns(v: float) -> float:
        try:
            return float(v) / 1e6
        except Exception:
            return 0.0

    total_cuda_ms = sum(to_ms_ns(e.get("cuda_time_total", 0.0)) for e in events)
    total_cpu_ms = sum(to_ms_ns(e.get("cpu_time_total", 0.0)) for e in events)
    by_time = sorted(events, key=lambda e: to_ms_ns(e.get("cuda_time_total", 0.0)), reverse=True)[:15]
    top_by_time = [
        {
            "name": e.get("name", ""),
            "cuda_ms": to_ms_ns(e.get("cuda_time_total", 0.0)),
            "cpu_ms": to_ms_ns(e.get("cpu_time_total", 0.0)),
            "count": int(e.get("count", 1)),
            "cuda_mem": int(e.get("self_cuda_memory_usage", 0)),
            "cpu_mem": int(e.get("self_cpu_memory_usage", 0)),
        }
        for e in by_time
    ]
    by_mem = sorted(events, key=lambda e: int(e.get("self_cuda_memory_usage", 0)), reverse=True)[:15]
    top_by_mem = [
        {
            "name": e.get("name", ""),
            "cuda_mem": int(e.get("self_cuda_memory_usage", 0)),
            "cpu_mem": int(e.get("self_cpu_memory_usage", 0)),
        }
        for e in by_mem
    ]
    return {
        "num_events": len(events),
        "total_cuda_time_ms": total_cuda_ms,
        "total_cpu_time_ms": total_cpu_ms,
        "top_by_time": top_by_time,
        "top_by_mem": top_by_mem,
    }


def run(model_path: Path, input_shape: Sequence[int], telemetry_out: Path, artifacts_dir: Path) -> int:
    # Quiet third-party libraries (TF/Gym) to avoid distracting stderr noise
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_LOGGING_MIN_LEVEL", "3")
    os.environ.setdefault("GYM_DISABLE_WARNINGS", "1")
    warnings.filterwarnings("ignore")
    logging.getLogger("absl").setLevel(logging.ERROR)

    print(f"{BLUE}›{RESET} Loading model: {model_path}")
    try:
        import torch
    except Exception as exc:
        print(f"{RED}✖{RESET} PyTorch not available: {exc}")
        return 1

    if not model_path.exists():
        print(f"{RED}✖{RESET} Model file not found: {model_path}")
        return 1

    # Try TorchScript first, then fall back to eager pickled nn.Module
    model = None
    load_error = None
    try:
        model = torch.jit.load(str(model_path))
    except Exception as exc:
        load_error = exc
    if model is None:
        try:
            # PyTorch >=2.6 defaults to weights_only=True which blocks arbitrary pickles
            # We override to False in this controlled environment to load the eager module
            model = torch.load(str(model_path), map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only argument
            model = torch.load(str(model_path), map_location="cpu")
        except Exception as exc2:
            print(f"{RED}✖{RESET} Failed to load model (jit: {load_error}, eager: {exc2})")
            return 1
    model = model.eval()
    print(f"{GREEN}✔{RESET} Model loaded")

    # Build a dummy input and prefer GPU placement
    x = torch.randn(*input_shape)
    if torch.cuda.is_available():
        try:
            model = model.to("cuda")
            x = x.to("cuda")
        except Exception:
            print(f"{YELLOW}!{RESET} Could not move model to CUDA; CPU profiling only")
    else:
        print(f"{YELLOW}!{RESET} CUDA not available; GPU timings will be 0")

    # Register progress hooks per-layer
    print(f"{BLUE}›{RESET} Registering layer progress hooks...")
    handles, layer_logs = _register_progress_hooks(model)
    if handles:
        print(f"{GREEN}✔{RESET} Hooks active on {len(handles)//2} modules")
    else:
        # For ScriptModules, Python hooks may be unavailable; wrap to ensure at least one hook fires
        class _Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, *a, **kw):  # type: ignore[no-untyped-def]
                return self.inner(*a, **kw)
        try:
            wrapped = _Wrapper(model)
            handles, layer_logs = _register_progress_hooks(wrapped)
            model = wrapped
            print(f"{GREEN}✔{RESET} Hooks active via wrapper")
        except Exception:
            print(f"{YELLOW}!{RESET} Hooks unavailable (continuing)")

    # Ensure artifacts directory exists
    _ensure_dir(artifacts_dir)

    # Profile with telemetry collector (torch.profiler + NVML)
    profile_model = None
    # Try installed package first
    try:
        from agnitra.telemetry_collector import profile_model as _pm  # type: ignore
        profile_model = _pm
    except Exception:
        pass
    if profile_model is None:
        # Fallback: load module directly from source file to avoid package import side-effects
        try:
            import importlib.util as _ilu
            from pathlib import Path as _P
            _tele_path = _P(__file__).resolve().parents[1] / "agnitra" / "telemetry_collector.py"
            spec = _ilu.spec_from_file_location("agnitra_telemetry_collector", str(_tele_path))
            assert spec and spec.loader
            mod = _ilu.module_from_spec(spec)
            import sys as _sys
            _sys.modules[spec.name] = mod  # ensure module is registered for decorators
            spec.loader.exec_module(mod)  # type: ignore[arg-type]
            profile_model = getattr(mod, "profile_model")
        except Exception as _exc:
            raise ImportError(f"Failed to import telemetry_collector: {_exc}")
    # Friendly device banner
    try:
        dev_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
    except Exception:
        dev_name = "CPU"
    print(f"{BLUE}›{RESET} Profiling (inference) on {dev_name} with torch.profiler + NVML...")
    try:
        payload = profile_model(model, x, str(telemetry_out), with_rl=False)
        print(f"{GREEN}✔{RESET} Inference telemetry written: {telemetry_out}")
    except Exception as exc:
        print(f"{RED}✖{RESET} Profiling failed: {exc}")
        return 1
    finally:
        # Remove hooks
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    # Persist layer log
    try:
        layer_log_path = artifacts_dir / f"layer_log_{model_path.stem}.json"
        layer_log_serialized = [
            {"name": lg.name, "elapsed_ms": lg.elapsed_ms} for _, lg in layer_logs.items() if lg.elapsed_ms is not None
        ]
        layer_log_path.write_text(json.dumps(layer_log_serialized, indent=2), encoding="utf-8")
        print(f"{GREEN}✔{RESET} Layer progress log saved: {layer_log_path}")
    except Exception:
        print(f"{YELLOW}!{RESET} Could not save layer progress log")

    # Additional: tiny training + validation steps to capture more telemetry
    try:
        from torch import nn, optim
        train_steps = 3
        print(f"{BLUE}›{RESET} Running training for {train_steps} steps...")
        # Define a simple loss on the model output
        model.train()
        x_train = torch.randn(*input_shape).to(x.device)
        y = torch.randn_like(model(x_train)).detach()
        criterion = nn.MSELoss()
        optimizer = optim.SGD(getattr(model, 'parameters', lambda: [])(), lr=1e-3) if hasattr(model, 'parameters') else None
        # Fallback: if parameters unavailable (pure ScriptModule), skip optimizer
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU] + (
                [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []
            ),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(train_steps):
                if optimizer:
                    optimizer.zero_grad(set_to_none=True)
                out = model(x_train)
                loss = criterion(out, y)
                loss.backward()
                if optimizer:
                    optimizer.step()
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
        # Serialize training events similarly to inference payload
        train_events = []
        for evt in prof.key_averages():
            cnt = int(getattr(evt, "count", 1))
            cpu_total = float(getattr(evt, "cpu_time_total", 0.0))
            cuda_total = float(getattr(evt, "cuda_time_total", 0.0))
            train_events.append({
                "name": str(evt.key),
                "cuda_time_total": cuda_total,
                "cpu_time_total": cpu_total,
                "self_cuda_memory_usage": int(getattr(evt, "self_cuda_memory_usage", 0)),
                "self_cpu_memory_usage": int(getattr(evt, "self_cpu_memory_usage", 0)),
                "count": cnt,
                "cuda_time_avg": (cuda_total / max(1, cnt)),
                "cpu_time_avg": (cpu_total / max(1, cnt)),
            })
        train_payload = {"events": train_events}
        telemetry_train_out = telemetry_out.with_name(telemetry_out.stem + "_train" + telemetry_out.suffix)
        telemetry_train_out.write_text(json.dumps(train_payload, indent=2), encoding="utf-8")
        print(f"{GREEN}✔{RESET} Training telemetry written: {telemetry_train_out}")

        # Validation (evaluation) pass for a few steps
        try:
            val_steps = 2
            print(f"{BLUE}›{RESET} Running validation for {val_steps} steps...")
            model.eval()
            x_val = torch.randn(*input_shape).to(x.device)
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU] + (
                    [torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else []
                ),
                record_shapes=True,
                profile_memory=True,
            ) as vprof:
                with torch.no_grad():
                    for _ in range(val_steps):
                        _ = model(x_val)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
            val_events = []
            for evt in vprof.key_averages():
                cnt = int(getattr(evt, "count", 1))
                cpu_total = float(getattr(evt, "cpu_time_total", 0.0))
                cuda_total = float(getattr(evt, "cuda_time_total", 0.0))
                val_events.append({
                    "name": str(evt.key),
                    "cuda_time_total": cuda_total,
                    "cpu_time_total": cpu_total,
                    "self_cuda_memory_usage": int(getattr(evt, "self_cuda_memory_usage", 0)),
                    "self_cpu_memory_usage": int(getattr(evt, "self_cpu_memory_usage", 0)),
                    "count": cnt,
                    "cuda_time_avg": (cuda_total / max(1, cnt)),
                    "cpu_time_avg": (cpu_total / max(1, cnt)),
                })
            telemetry_val_out = telemetry_out.with_name(telemetry_out.stem + "_val" + telemetry_out.suffix)
            telemetry_val_out.write_text(json.dumps({"events": val_events}, indent=2), encoding="utf-8")
            print(f"{GREEN}✔{RESET} Validation telemetry written: {telemetry_val_out}")
        except Exception as exc:
            print(f"{YELLOW}!{RESET} Validation telemetry skipped: {exc}")
    except Exception as exc:
        print(f"{YELLOW}!{RESET} Training telemetry skipped: {exc}")

    # Create result summary artifact
    try:
        events = payload.get("events", []) if isinstance(payload, dict) else []
        summary = _summarize(events)
        # Include training/validation summaries if available
        try:
            tpath = telemetry_out.with_name(telemetry_out.stem + "_train" + telemetry_out.suffix)
            if tpath.exists():
                tdata = json.loads(tpath.read_text(encoding="utf-8"))
                tsummary = _summarize(tdata.get("events", []))
            else:
                tsummary = None
        except Exception:
            tsummary = None
        try:
            vpath = telemetry_out.with_name(telemetry_out.stem + "_val" + telemetry_out.suffix)
            if vpath.exists():
                vdata = json.loads(vpath.read_text(encoding="utf-8"))
                vsummary = _summarize(vdata.get("events", []))
            else:
                vsummary = None
        except Exception:
            vsummary = None
        result = {
            "model": model_path.name,
            "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"),
            "meta": payload.get("meta", {}),
            "gpu": payload.get("gpu", {}),
            "behavior": payload.get("behavior", {}),
            "opportunities": payload.get("opportunities", {}),
            "summary": summary,
            # Include rich per-layer profile (params, buffers, IO shapes, dtypes)
            "model_profile": payload.get("model", {}),
            "training_summary": tsummary,
            "validation_summary": vsummary,
        }
        result_path = artifacts_dir / f"profile_result_{model_path.stem}.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"{GREEN}✔{RESET} Result summary saved: {result_path}")
    except Exception as exc:
        print(f"{YELLOW}!{RESET} Failed to write result summary: {exc}")

    print(f"{GREEN}✔{RESET} Profiling completed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a TorchScript model with rich telemetry")
    parser.add_argument("model", type=Path, help="Path to TorchScript model (.pt)")
    parser.add_argument("--input-shape", default="1,3,224,224", help="Comma separated input shape, e.g. 1,3,224,224")
    parser.add_argument("--output", default="telemetry.json", help="Telemetry JSON output path")
    parser.add_argument("--artifacts-dir", default=str(Path("agnitraai")/"context"), help="Directory for result artifacts")
    args = parser.parse_args(argv)

    return run(
        model_path=args.model,
        input_shape=_parse_shape(args.input_shape),
        telemetry_out=Path(args.output),
        artifacts_dir=Path(args.artifacts_dir),
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
