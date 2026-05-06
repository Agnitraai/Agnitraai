"""The :class:`InferenceManifest` schema and builder.

A manifest is a tamper-evident record of what Agnitra did to a model.
It serializes to canonical JSON for signing and to a regular dict for
inclusion in :class:`agnitra.core.runtime.RuntimeOptimizationResult`'s
``notes``.

Schema versioning: the ``version`` field is checked at verification
time; ``agnitra-trust/v1`` is the only currently-defined version.
Future versions will document migration paths in CHANGELOG.md.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

LOGGER = logging.getLogger(__name__)

MANIFEST_VERSION = "agnitra-trust/v1"


@dataclass
class BaseModelClaim:
    """Identifies the input model the optimizer started from."""
    sha256: str
    fingerprint_signature: str   # short architecture sig from agnitra.core.runtime.fingerprint
    fingerprint_dict: Dict[str, Any]
    param_count: int
    source: Optional[str] = None        # e.g. "huggingface://meta-llama/Meta-Llama-3-8B"
    license: Optional[str] = None
    detected_architecture: Optional[str] = None


@dataclass
class OptimizationStep:
    """One pass that ran during optimize()."""
    name: str                           # "tf32", "sdpa", "static_kv_cache", "compile", "quantize"
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None     # ISO8601 UTC


@dataclass
class VerificationMetrics:
    """Drift metrics from the post-optimization safety check."""
    cosine_similarity: Optional[float]
    argmax_match_rate: Optional[float]
    max_abs_diff: Optional[float]
    drift_passed: bool = True
    eval_suite: Optional[str] = None    # filled by Layer 3 (certified recipes)
    reverted: bool = False              # True if SDK fell back to baseline


@dataclass
class RuntimeContext:
    """The environment where optimize() ran."""
    torch_version: str
    cuda_version: Optional[str]
    agnitra_version: str
    hardware: Dict[str, Any]            # gpu_name, capability, memory_bytes
    hostname: Optional[str] = None


@dataclass
class SignerInfo:
    """Identity of the signing key. Filled by sign_manifest().

    ``public_key`` and ``key_id`` are empty until the manifest is signed.
    A manifest with empty signer info is unsigned and verify_manifest()
    will reject it.
    """
    public_key: str = ""                # "ed25519:" + base64(raw 32 bytes)
    key_id: str = ""                    # 16-char hex prefix of SHA-256(public key bytes)


@dataclass
class InferenceManifest:
    """A signed-or-unsigned record of one optimize() call.

    The ``signature`` field is intentionally OUTSIDE the canonical bytes
    that get signed — otherwise we'd have a chicken-and-egg problem.
    See :meth:`canonical_bytes` for the exact serialization rules.
    """
    version: str
    issued_at: str                      # ISO8601 UTC, second precision
    base_model: BaseModelClaim
    optimizations: List[OptimizationStep]
    verification: VerificationMetrics
    runtime: RuntimeContext
    signer: SignerInfo = field(default_factory=SignerInfo)
    signature: Optional[str] = None

    def canonical_bytes(self) -> bytes:
        """Bytes that get signed.

        Rules:
        * ``json.dumps`` with ``sort_keys=True`` and tight separators
          (no whitespace).
        * The ``signature`` field is excluded.
        * Everything else, including the signer's public key, IS included
          (so swapping the key invalidates the signature).
        """
        d = asdict(self)
        d.pop("signature", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")

    def to_dict(self) -> Dict[str, Any]:
        """Plain JSON-serializable dict, including the signature when set."""
        return asdict(self)


# --------------------------------------------------------------------------
# Builder — converts a RuntimeOptimizationResult into an unsigned manifest
# --------------------------------------------------------------------------

def _utc_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    if isinstance(d, Mapping) and key in d:
        return d[key]
    return default


def _agnitra_version() -> str:
    try:
        from importlib import metadata
        return metadata.version("agnitra")
    except Exception:
        return "0.0.0"


def _runtime_context(hardware: Optional[Dict[str, Any]] = None) -> RuntimeContext:
    """Capture the current runtime — torch / CUDA / GPU / agnitra version."""
    torch_version = "unknown"
    cuda_version = None
    try:
        import torch
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    except Exception:
        pass

    if hardware is None:
        try:
            from agnitra.core.runtime.fingerprint import _gpu_fingerprint  # type: ignore
            hardware = _gpu_fingerprint()
        except Exception:
            hardware = {}

    hostname = None
    try:
        import socket
        hostname = socket.gethostname()
    except Exception:  # pragma: no cover - defensive
        pass

    return RuntimeContext(
        torch_version=torch_version,
        cuda_version=cuda_version,
        agnitra_version=_agnitra_version(),
        hardware=dict(hardware) if hardware else {},
        hostname=hostname,
    )


def build_manifest_from_result(
    *,
    base_model: Any,
    optimized_model: Any,           # noqa: ARG001 - reserved for future use
    agnitra_result: Any,            # RuntimeOptimizationResult
    quantize_mode: Optional[str] = None,
    source: Optional[str] = None,
    license_label: Optional[str] = None,
) -> InferenceManifest:
    """Build an unsigned :class:`InferenceManifest` from a SDK result.

    The caller is expected to immediately pass this to
    :func:`agnitra.trust.sign_manifest` to get a signed copy.

    Params:
        base_model: the unmodified ``nn.Module`` (used for SHA + fingerprint)
        optimized_model: the post-optimize module (currently unused, reserved
            for future "diff-of-weights" claims)
        agnitra_result: the ``RuntimeOptimizationResult`` returned by
            ``agnitra.optimize`` — used to extract optimizations,
            validation metrics, and architecture detection
        quantize_mode: the ``quantize`` argument passed to ``optimize()``;
            included as the only extant optimization parameter we can
            recover from the call site (the rest are inferred from
            ``apply_universal``'s known sequence)
        source: optional URI the model came from (HF Hub, local path, etc.)
        license_label: optional license string; if ``None`` we attempt
            to read it from ``base_model.config.license`` and fall back
            to ``None``

    Never raises; on any failure to extract a field, that field becomes
    ``None`` / ``0`` / empty so the manifest is at least well-formed.
    """
    from agnitra.trust.digest import model_sha256

    # Architecture identity
    fingerprint_signature = ""
    fingerprint_dict: Dict[str, Any] = {}
    try:
        from agnitra.core.runtime.fingerprint import (
            architecture_fingerprint,
            architecture_signature,
        )
        fingerprint_dict = architecture_fingerprint(base_model)
        fingerprint_signature = architecture_signature(fingerprint_dict)
    except Exception:
        LOGGER.exception("Failed to compute architecture fingerprint for manifest")

    # Param count
    param_count = 0
    try:
        param_count = int(sum(p.numel() for p in base_model.parameters()))
    except Exception:  # pragma: no cover - defensive against non-torch models
        pass

    # License — best-effort lookup from model.config.license
    if license_label is None:
        try:
            cfg = getattr(base_model, "config", None)
            if cfg is not None:
                license_label = getattr(cfg, "license", None)
        except Exception:
            license_label = None

    # Detected architecture
    notes = getattr(agnitra_result, "notes", {}) or {}
    detected_arch = _safe_get(notes, "detected_architecture")

    base_claim = BaseModelClaim(
        sha256=model_sha256(base_model, incremental=True),
        fingerprint_signature=fingerprint_signature,
        fingerprint_dict=fingerprint_dict,
        param_count=param_count,
        source=source,
        license=license_label,
        detected_architecture=detected_arch,
    )

    # Optimization steps — for ring-1 specialists we know the universal
    # sequence applied. For the legacy agent path, we only know what the
    # notes report. Either way, the manifest reflects what was attempted.
    timestamp = _utc_iso()
    optimizations: List[OptimizationStep] = []
    if notes.get("specialist"):
        for name in ("tf32", "sdpa", "static_kv_cache", "compile"):
            optimizations.append(OptimizationStep(name=name, timestamp=timestamp))
    if quantize_mode:
        optimizations.append(OptimizationStep(
            name="quantize",
            parameters={"mode": quantize_mode},
            timestamp=timestamp,
        ))

    # Verification metrics
    val = _safe_get(notes, "validation", {}) or {}
    verification = VerificationMetrics(
        cosine_similarity=_safe_get(val, "cosine_similarity"),
        argmax_match_rate=_safe_get(val, "argmax_match_rate"),
        max_abs_diff=_safe_get(val, "max_abs_diff"),
        drift_passed=not bool(_safe_get(val, "regressed", False)),
        reverted=bool(notes.get("reverted_due_to_drift", False)),
    )

    return InferenceManifest(
        version=MANIFEST_VERSION,
        issued_at=timestamp,
        base_model=base_claim,
        optimizations=optimizations,
        verification=verification,
        runtime=_runtime_context(),
        signer=SignerInfo(),
        signature=None,
    )


__all__ = [
    "MANIFEST_VERSION",
    "BaseModelClaim",
    "InferenceManifest",
    "OptimizationStep",
    "RuntimeContext",
    "SignerInfo",
    "VerificationMetrics",
    "build_manifest_from_result",
]
