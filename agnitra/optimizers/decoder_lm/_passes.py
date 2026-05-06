"""Decoder-LM optimization passes — proven, hard-coded.

Each function takes a model (and possibly other args) and returns the
transformed model. Functions are pure-ish: they may mutate global torch
config (e.g. TF32 flags) or model attributes (e.g. set
``cache_implementation="static"`` on ``model.config``), but they do
not change weights.

The passes here are the ones we *know* work for decoder-only LLMs:

* TF32 matmul on Ampere+
* SDPA attention (PyTorch 2.x's default flash-attention path)
* Static KV cache pre-allocation
* ``torch.compile`` with reduce-overhead mode (CUDA graphs)

Architecture-specific fusions (RoPE, RMSNorm, etc.) live in the
per-architecture modules and call into these primitives plus their own
custom transforms.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

LOGGER = logging.getLogger(__name__)


def enable_tf32() -> None:
    """Enable TF32 matmul on Ampere+ GPUs.

    A free 1.3-1.6x on H100 matmuls; tiny precision impact (mantissa
    truncated from 23 to 10 bits, exponent unchanged). Safe for
    inference because attention uses bf16/fp16 anyway.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def ensure_sdpa_attention(model: Any) -> Any:
    """Verify (and best-effort enable) SDPA attention on HF transformers.

    PyTorch 2.x's scaled_dot_product_attention dispatches to FlashAttention-2
    on H100/A100. HuggingFace models default to "eager" attention pre-4.42
    and "sdpa" on later versions — we explicitly set it when supported so
    older transformers versions still get the fused kernel.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    current = getattr(cfg, "_attn_implementation", None)
    if current == "sdpa":
        return model
    # Some HF model classes accept _attn_implementation; setting it on
    # config alone is a no-op after construction. Try both.
    try:
        cfg._attn_implementation = "sdpa"
    except Exception:  # pragma: no cover - defensive
        LOGGER.debug("Could not set _attn_implementation='sdpa' on config")
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("sdpa")
        except Exception:  # pragma: no cover
            pass
    return model


def enable_static_kv_cache(model: Any) -> Any:
    """Set ``config.cache_implementation = "static"`` for HF generation.

    Static KV cache pre-allocates the full sequence-length buffer up
    front, eliminating per-step allocator pressure and unlocking
    cudagraphs in ``model.generate()``. Requires HF transformers 4.38+.
    """
    cfg = getattr(model, "config", None)
    if cfg is None:
        return model
    try:
        cfg.cache_implementation = "static"
    except Exception:  # pragma: no cover
        LOGGER.debug("Could not set cache_implementation='static' on config")
    return model


def compile_for_decode(model: Any) -> Any:
    """Wrap the model in ``torch.compile(mode='reduce-overhead')``.

    ``reduce-overhead`` reuses CUDA graphs across calls — best mode for
    autoregressive decode at fixed shapes. Requires the static KV cache
    to be set first; otherwise dynamic-shape recompilations defeat
    the cudagraph reuse and the mode degrades to ``default``.

    We use ``fullgraph=False`` because HF generation has Python-side
    branches (sampling, stopping criteria) that won't trace cleanly.
    The fast path is the inner forward.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover
        return model
    try:
        return torch.compile(model, mode="reduce-overhead", fullgraph=False)
    except Exception as exc:
        LOGGER.warning("torch.compile failed (%s); returning uncompiled model", exc)
        return model


# --- Sequence builders ----------------------------------------------------


def apply_universal(
    model: Any,
    *,
    sample_input: Any,
    enable_compile: bool = True,
    quantize: Optional[str] = None,
) -> Any:
    """The universal-decoder-LM sequence: TF32 + SDPA + static cache + compile.

    Applied to every ring-1 architecture. Architecture-specialist
    modules call this first, then layer their own passes on top.

    ``quantize`` accepts:
      * ``"int8_weight"`` (W8A16): ~2x memory, ~1.3-1.7x throughput,
        near-zero quality loss. Default safe choice; works on any CUDA GPU.
      * ``"int4_weight"`` (W4A16): ~4x memory, ~1.6-2.0x throughput.
        Mild quality drop; verify against your eval set.
      * ``"fp8_weight"`` (W8(FP8)A8(FP8)): ~2x throughput on H100 /
        Blackwell tensor cores, near-zero quality drop. Falls back to
        slower emulation on pre-Hopper GPUs — use ``"auto"`` to avoid
        that footgun.
      * ``"auto"``: picks FP8 on H100+/Blackwell, INT8 elsewhere.
      * ``None`` (default): no quantization.

    Quantization is applied BEFORE compile so torch.compile captures
    the dequantize+matmul kernel pattern under cudagraphs.
    """
    enable_tf32()
    model = ensure_sdpa_attention(model)
    model = enable_static_kv_cache(model)
    # Quantize BEFORE compile: torch.compile then captures the
    # dequantize+matmul kernel pattern under cudagraphs.
    if quantize is not None:
        from agnitra.optimizers.decoder_lm._quantization import apply_quantization
        model = apply_quantization(model, mode=quantize)
    if enable_compile:
        model = compile_for_decode(model)
    return model


__all__ = [
    "enable_tf32",
    "ensure_sdpa_attention",
    "enable_static_kv_cache",
    "compile_for_decode",
    "apply_universal",
]
