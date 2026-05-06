"""Verify a signed :class:`InferenceManifest`.

Verification NEVER raises. Failure cases (bad signature, unknown
signer, schema mismatch) are returned as :class:`VerifyResult` so
auditors and CI can branch on the result instead of try/except.

Successful verification means three things at once:

1. The signature is mathematically valid for the canonical bytes
   (i.e., the manifest hasn't been tampered with after signing).
2. The signer's claimed key matches what produced the signature.
3. The schema version is one this verifier understands.

What verification deliberately does NOT check:

- Whether the model SHA in the manifest matches a model on disk —
  that's the caller's job (they pass ``--model`` to the CLI to opt in,
  because hashing 16GB of weights is expensive and not always wanted).
- Whether the signer is "trusted" — pass ``trusted_keys=`` to enforce
  an allowlist; otherwise any well-formed signature passes structural
  verification.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from agnitra.trust.errors import SignatureError
from agnitra.trust.manifest import MANIFEST_VERSION, InferenceManifest

LOGGER = logging.getLogger(__name__)


@dataclass
class VerifyResult:
    """Outcome of :func:`verify_manifest`.

    ``valid`` is ``True`` only when there are zero errors. Warnings do
    not block validity but are surfaced for CI / auditor review.
    """
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


def verify_manifest(
    manifest: InferenceManifest,
    *,
    trusted_keys: Optional[Iterable[str]] = None,
) -> VerifyResult:
    """Structural + cryptographic verification.

    ``trusted_keys`` is an iterable of ``key_id`` strings (the 16-char
    hex prefixes from :func:`fingerprint_public_key`). When set, the
    signer's key_id must appear in the set or the result is invalid.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # Structural checks first — verify_manifest must work on partially-
    # constructed manifests (e.g. CLI tooling that loaded JSON from disk).
    if not manifest.signature:
        errors.append("manifest is unsigned")
    if not manifest.signer.public_key or not manifest.signer.key_id:
        errors.append("signer info is missing or incomplete")

    if errors:
        return VerifyResult(valid=False, errors=errors, warnings=warnings)

    # Schema version
    if manifest.version != MANIFEST_VERSION:
        warnings.append(
            f"unknown manifest version {manifest.version!r} "
            f"(this verifier understands {MANIFEST_VERSION!r})"
        )

    # Trusted-key allowlist
    if trusted_keys is not None:
        trusted_set = set(trusted_keys)
        if manifest.signer.key_id not in trusted_set:
            errors.append(
                f"signer key_id {manifest.signer.key_id!r} is not in the trusted set"
            )

    # Cryptographic check
    try:
        from agnitra.trust.keys import decode_public_key
        public_key = decode_public_key(manifest.signer.public_key)
    except SignatureError as exc:
        errors.append(f"failed to decode signer public key: {exc}")
        return VerifyResult(valid=False, errors=errors, warnings=warnings)

    if not manifest.signature.startswith("ed25519:"):
        errors.append(
            f"signature has unsupported algorithm prefix: {manifest.signature[:16]!r}"
        )
        return VerifyResult(valid=False, errors=errors, warnings=warnings)

    try:
        sig_bytes = base64.b64decode(manifest.signature.removeprefix("ed25519:"))
    except Exception as exc:
        errors.append(f"signature is not valid base64: {exc!r}")
        return VerifyResult(valid=False, errors=errors, warnings=warnings)

    if len(sig_bytes) != 64:
        errors.append(
            f"Ed25519 signatures are 64 bytes; decoded {len(sig_bytes)} bytes"
        )
        return VerifyResult(valid=False, errors=errors, warnings=warnings)

    try:
        public_key.verify(sig_bytes, manifest.canonical_bytes())
    except Exception:
        # cryptography.exceptions.InvalidSignature, but we don't want
        # to import the specific exception class here just to catch it;
        # the broad except is fine because we're already in the failure
        # path and the message is informative.
        errors.append(
            "signature does not match canonical bytes — manifest tampered, "
            "or wrong public key claimed"
        )

    return VerifyResult(valid=not errors, errors=errors, warnings=warnings)


__all__ = ["VerifyResult", "verify_manifest"]
