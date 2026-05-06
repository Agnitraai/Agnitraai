"""Cryptographic trust and provenance layer for Agnitra AI.

Layer 1 of the trust roadmap: signed inference manifests. Every
``agnitra.optimize()`` call produces a tamper-evident record of:

* the base model's deterministic SHA-256 over its weights
* its architecture fingerprint (the cross-fine-tune cache key)
* every optimization step that ran (TF32, SDPA, static cache,
  quantization mode, torch.compile)
* drift verification metrics (cosine sim, argmax match rate)
* the runtime context (torch / CUDA / hardware UUID / agnitra version)
* the signer's public key + key fingerprint
* an Ed25519 signature over the canonical bytes of all of the above

Why this exists: regulated deployments (banking, healthcare, EU AI Act
high-risk systems) need cryptographic proof of which model was
optimized, with what, and that the result was validated. ``ed25519``
signatures over canonical JSON give auditors something they can verify
without trusting any single vendor.

Layers 2-5 (per-inference provenance tags, certified quantization
recipes, cross-runtime determinism, ZK proofs) build on this foundation
but ship in subsequent releases.

Public API:

* :class:`InferenceManifest` — the dataclass schema
* :func:`sign_manifest` — produce an Ed25519 signature
* :func:`verify_manifest` — never raises; returns a structured
  :class:`VerifyResult`
* :func:`build_manifest_from_result` — convert a SDK
  ``RuntimeOptimizationResult`` into an unsigned manifest

The cryptography dependency is optional: ``pip install "agnitra[trust]"``.
"""
from __future__ import annotations

from agnitra.trust.errors import (
    ManifestError,
    SignatureError,
    TrustError,
)
from agnitra.trust.manifest import (
    BaseModelClaim,
    InferenceManifest,
    OptimizationStep,
    RuntimeContext,
    SignerInfo,
    VerificationMetrics,
    build_manifest_from_result,
)
from agnitra.trust.sign import sign_manifest
from agnitra.trust.verify import VerifyResult, verify_manifest
from agnitra.trust.digest import model_sha256

__all__ = [
    "BaseModelClaim",
    "InferenceManifest",
    "ManifestError",
    "OptimizationStep",
    "RuntimeContext",
    "SignatureError",
    "SignerInfo",
    "TrustError",
    "VerificationMetrics",
    "VerifyResult",
    "build_manifest_from_result",
    "model_sha256",
    "sign_manifest",
    "verify_manifest",
]
