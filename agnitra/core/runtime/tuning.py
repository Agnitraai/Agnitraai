"""Apply LLM/agent tuning presets to runtime and model.

This module provides best-effort helpers to enact hardware-friendly tuning
suggestions returned by LLMs or RL agents. All operations degrade gracefully
when features are unavailable (e.g., CPU-only environments).
"""

from __future__ import annotations

from typing import Any, Dict

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None  # type: ignore


def apply_tuning_preset(model: Any, preset: Dict[str, Any]) -> Any:
    """Apply a tuning preset to PyTorch backends and optionally wrap the model.

    Supported keys in ``preset``:
      - ``allow_tf32`` (bool): enables TF32 matmul on Ampere+ GPUs.
      - ``flash_sdp`` (bool): prefer FlashAttention/SDPA kernels if available.
      - ``torch_compile`` (bool): compile the model if PyTorch 2.x is present.
      - ``kv_cache_dtype`` (str): advisory only; recorded as a model attribute.

    Returns the possibly-wrapped ``model``.
    """

    if torch is None:
        return model

    allow_tf32 = bool(preset.get("allow_tf32", False))
    flash_sdp = bool(preset.get("flash_sdp", False))
    use_compile = bool(preset.get("torch_compile", False))
    kv_cache_dtype = str(preset.get("kv_cache_dtype", "")).lower() or None

    # TF32 matmul toggle (safe no-op on older GPUs)
    try:  # pragma: no cover - backend attribute may vary by version
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
    except Exception:
        pass

    # SDPA backend preferences (PyTorch 2.x)
    try:  # pragma: no cover - API availability dependent
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(bool(flash_sdp))
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            # keep mem-efficient enabled when flash is on; disable only if both off
            torch.backends.cuda.enable_mem_efficient_sdp(bool(flash_sdp))
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            # keep math fallback available
            torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    # Advisory: record KV cache dtype for downstream components to consume
    try:
        if kv_cache_dtype:
            setattr(model, "_agnitra_kv_cache_dtype", kv_cache_dtype)
    except Exception:
        pass

    # Optionally compile the model (PyTorch 2.x).
    try:  # pragma: no cover - torch.compile not available in all envs
        if use_compile and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[call-arg]
    except Exception:
        # Compilation may fail on unsupported ops; keep baseline model
        pass

    return model


__all__ = ["apply_tuning_preset"]

