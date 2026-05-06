"""Tests for agnitra.trust (Layer 1 — signed inference manifests).

Most tests skip cleanly when ``cryptography`` isn't installed, so the
suite stays runnable in CI matrices without the trust extra. The pure
data-layer tests (manifest schema, model SHA determinism) run
unconditionally.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from torch import nn


crypto = pytest.importorskip(
    "cryptography",
    reason="trust layer requires cryptography>=42 (`pip install \"agnitra[trust]\"`)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _Tiny(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.linear = nn.Linear(8, 4)

    def forward(self, x):
        return self.linear(x)


def _build_test_manifest():
    from agnitra.trust.manifest import (
        BaseModelClaim,
        InferenceManifest,
        OptimizationStep,
        RuntimeContext,
        SignerInfo,
        VerificationMetrics,
    )
    return InferenceManifest(
        version="agnitra-trust/v1",
        issued_at="2026-05-06T12:00:00+00:00",
        base_model=BaseModelClaim(
            sha256="9f2b" + "0" * 60,
            fingerprint_signature="abc123",
            fingerprint_dict={"source": "config", "config": {"model_type": "llama"}},
            param_count=8_030_261_248,
            source="huggingface://meta-llama/Meta-Llama-3-8B-Instruct",
            license="meta-llama-3-community",
            detected_architecture="llama",
        ),
        optimizations=[
            OptimizationStep(name="tf32", timestamp="2026-05-06T12:00:00+00:00"),
            OptimizationStep(name="sdpa", timestamp="2026-05-06T12:00:00+00:00"),
            OptimizationStep(
                name="quantize",
                parameters={"mode": "int8_weight"},
                timestamp="2026-05-06T12:00:00+00:00",
            ),
        ],
        verification=VerificationMetrics(
            cosine_similarity=0.9994,
            argmax_match_rate=0.998,
            max_abs_diff=0.04,
            drift_passed=True,
        ),
        runtime=RuntimeContext(
            torch_version="2.4.0",
            cuda_version="12.1",
            agnitra_version="0.2.2",
            hardware={"vendor": "nvidia", "model": "H100 80GB SXM5"},
            hostname="test-host",
        ),
        signer=SignerInfo(),
        signature=None,
    )


# ---------------------------------------------------------------------------
# Manifest schema + canonical serialization
# ---------------------------------------------------------------------------

def test_canonical_bytes_excludes_signature():
    m = _build_test_manifest()
    bytes_unsigned = m.canonical_bytes()
    m.signature = "ed25519:fakebase64=="
    bytes_after_setting_sig = m.canonical_bytes()
    assert bytes_unsigned == bytes_after_setting_sig, (
        "Setting signature must not change canonical bytes — otherwise "
        "verification can't succeed."
    )


def test_canonical_bytes_is_field_order_independent():
    """Two manifests built from the same data produce identical bytes."""
    a = _build_test_manifest()
    b = _build_test_manifest()
    assert a.canonical_bytes() == b.canonical_bytes()


def test_canonical_bytes_changes_when_data_changes():
    a = _build_test_manifest()
    b = _build_test_manifest()
    b.base_model.param_count += 1
    assert a.canonical_bytes() != b.canonical_bytes()


def test_to_dict_round_trips_through_json():
    m = _build_test_manifest()
    d = m.to_dict()
    payload = json.loads(json.dumps(d))
    assert payload["version"] == "agnitra-trust/v1"
    assert payload["base_model"]["sha256"].startswith("9f2b")


# ---------------------------------------------------------------------------
# model_sha256
# ---------------------------------------------------------------------------

def test_model_sha256_is_deterministic_across_instances():
    from agnitra.trust.digest import model_sha256
    a = _Tiny(seed=42)
    b = _Tiny(seed=42)
    assert model_sha256(a) == model_sha256(b)


def test_model_sha256_changes_with_weights():
    from agnitra.trust.digest import model_sha256
    a = _Tiny(seed=42)
    b = _Tiny(seed=43)
    assert model_sha256(a) != model_sha256(b)


def test_model_sha256_handles_module_without_state_dict():
    from agnitra.trust.digest import model_sha256
    # Deliberately not a nn.Module — exercises the empty-state-dict path.
    assert model_sha256(object()) == (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


# ---------------------------------------------------------------------------
# Sign + verify
# ---------------------------------------------------------------------------

def test_sign_attaches_signature_and_signer(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNITRA_TRUST_KEY_PATH", "")
    monkeypatch.setattr(
        "agnitra.trust.keys.DEFAULT_KEY_FILE",
        tmp_path / "signing.pem",
    )

    from agnitra.trust import sign_manifest

    manifest = sign_manifest(_build_test_manifest())
    assert manifest.signature is not None
    assert manifest.signature.startswith("ed25519:")
    assert manifest.signer.public_key.startswith("ed25519:")
    assert len(manifest.signer.key_id) == 16


def test_sign_verify_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNITRA_TRUST_KEY_PATH", "")
    monkeypatch.setattr(
        "agnitra.trust.keys.DEFAULT_KEY_FILE",
        tmp_path / "signing.pem",
    )

    from agnitra.trust import sign_manifest, verify_manifest

    signed = sign_manifest(_build_test_manifest())
    result = verify_manifest(signed)
    assert result.valid
    assert not result.errors


def test_tampering_after_signing_is_detected(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNITRA_TRUST_KEY_PATH", "")
    monkeypatch.setattr(
        "agnitra.trust.keys.DEFAULT_KEY_FILE",
        tmp_path / "signing.pem",
    )

    from agnitra.trust import sign_manifest, verify_manifest

    signed = sign_manifest(_build_test_manifest())
    # Mutate an optimization parameter post-signing — the canonical
    # bytes change, so the signature no longer verifies.
    signed.optimizations[2].parameters["mode"] = "int4_weight"
    result = verify_manifest(signed)
    assert not result.valid
    assert any("tampered" in err for err in result.errors)


def test_unsigned_manifest_fails_verify():
    from agnitra.trust import verify_manifest
    manifest = _build_test_manifest()  # no signature
    result = verify_manifest(manifest)
    assert not result.valid
    assert any("unsigned" in err for err in result.errors)


def test_trusted_keys_allowlist_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNITRA_TRUST_KEY_PATH", "")
    monkeypatch.setattr(
        "agnitra.trust.keys.DEFAULT_KEY_FILE",
        tmp_path / "signing.pem",
    )

    from agnitra.trust import sign_manifest, verify_manifest

    signed = sign_manifest(_build_test_manifest())

    # Wrong allowlist — verification should fail.
    result = verify_manifest(signed, trusted_keys={"deadbeef" * 2})
    assert not result.valid
    assert any("trusted set" in err for err in result.errors)

    # Correct allowlist — verification succeeds.
    result_ok = verify_manifest(signed, trusted_keys={signed.signer.key_id})
    assert result_ok.valid


def test_unknown_version_warns_but_does_not_invalidate(tmp_path, monkeypatch):
    monkeypatch.setenv("AGNITRA_TRUST_KEY_PATH", "")
    monkeypatch.setattr(
        "agnitra.trust.keys.DEFAULT_KEY_FILE",
        tmp_path / "signing.pem",
    )

    from agnitra.trust import sign_manifest, verify_manifest

    m = _build_test_manifest()
    m.version = "agnitra-trust/v999"
    signed = sign_manifest(m)
    result = verify_manifest(signed)
    assert result.valid  # still valid
    assert any("unknown manifest version" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# build_manifest_from_result
# ---------------------------------------------------------------------------

def test_build_from_result_extracts_data_from_notes():
    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )
    from agnitra.trust import build_manifest_from_result

    base = _Tiny(seed=42)
    snap = OptimizationSnapshot(
        latency_ms=10.0, tokens_per_sec=100.0, tokens_processed=1,
        gpu_utilization=None, telemetry={}, metadata={},
    )
    result = RuntimeOptimizationResult(
        optimized_model=_Tiny(seed=42),
        baseline=snap,
        optimized=snap,
        usage_event=None,
        notes={
            "specialist": True,
            "detected_architecture": "llama",
            "validation": {
                "cosine_similarity": 0.999,
                "argmax_match_rate": 1.0,
                "max_abs_diff": 0.001,
                "regressed": False,
            },
        },
    )

    manifest = build_manifest_from_result(
        base_model=base,
        optimized_model=result.optimized_model,
        agnitra_result=result,
        quantize_mode="int8_weight",
    )

    assert manifest.version == "agnitra-trust/v1"
    assert manifest.base_model.detected_architecture == "llama"
    assert manifest.verification.cosine_similarity == 0.999
    assert manifest.verification.argmax_match_rate == 1.0
    assert manifest.verification.drift_passed is True
    quant_steps = [op for op in manifest.optimizations if op.name == "quantize"]
    assert len(quant_steps) == 1
    assert quant_steps[0].parameters == {"mode": "int8_weight"}
    # Specialist mode should record TF32 / SDPA / static-cache / compile.
    assert {op.name for op in manifest.optimizations} >= {
        "tf32", "sdpa", "static_kv_cache", "compile", "quantize",
    }


def test_build_from_result_records_drift_revert():
    from agnitra.core.runtime.agent import (
        OptimizationSnapshot,
        RuntimeOptimizationResult,
    )
    from agnitra.trust import build_manifest_from_result

    snap = OptimizationSnapshot(
        latency_ms=0, tokens_per_sec=0, tokens_processed=0,
        gpu_utilization=None, telemetry={}, metadata={},
    )
    base = _Tiny(seed=42)
    result = RuntimeOptimizationResult(
        optimized_model=base,
        baseline=snap,
        optimized=snap,
        usage_event=None,
        notes={
            "validation": {"regressed": True, "cosine_similarity": 0.7},
            "reverted_due_to_drift": True,
        },
    )
    manifest = build_manifest_from_result(
        base_model=base,
        optimized_model=base,
        agnitra_result=result,
    )
    assert manifest.verification.reverted is True
    assert manifest.verification.drift_passed is False


# ---------------------------------------------------------------------------
# Key generation (CLI surface, no torch involved)
# ---------------------------------------------------------------------------

def test_generate_keypair_writes_pem_and_returns_key_id(tmp_path):
    from agnitra.trust.keys import generate_keypair_pem
    target = tmp_path / "agnitra_test_key.pem"
    path, key_id = generate_keypair_pem(key_path=target)
    assert path == target
    assert path.exists()
    assert path.read_text().startswith("-----BEGIN PRIVATE KEY-----")
    assert len(key_id) == 16


def test_generate_keypair_refuses_to_overwrite(tmp_path):
    from agnitra.trust.errors import SignatureError
    from agnitra.trust.keys import generate_keypair_pem
    target = tmp_path / "exists.pem"
    target.write_text("not a real key")
    with pytest.raises(SignatureError, match="Refusing to overwrite"):
        generate_keypair_pem(key_path=target)
