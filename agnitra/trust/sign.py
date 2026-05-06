"""Sign an :class:`InferenceManifest` with Ed25519.

Signing is destructive: the manifest's ``signer`` and ``signature``
fields are mutated in place AND the same instance is returned. Both
patterns work::

    sign_manifest(m)         # m is now signed
    m = sign_manifest(m)     # also fine

Why mutate-and-return: the SDK builds a manifest, hands it to
``sign_manifest``, and stuffs the result into
``RuntimeOptimizationResult.notes``. Forcing callers to rebind would
double the lines without buying anything.
"""
from __future__ import annotations

import base64
import logging
from typing import Any, Optional

from agnitra.trust.errors import SignatureError
from agnitra.trust.keys import (
    encode_public_key,
    fingerprint_public_key,
    load_or_create_signing_key,
)
from agnitra.trust.manifest import InferenceManifest, SignerInfo

LOGGER = logging.getLogger(__name__)


def sign_manifest(
    manifest: InferenceManifest,
    *,
    key: Optional[Any] = None,
) -> InferenceManifest:
    """Attach an Ed25519 signature to ``manifest``.

    The signing key resolution is the same as
    :func:`agnitra.trust.keys.load_or_create_signing_key` unless an
    explicit ``key`` (an ``Ed25519PrivateKey`` instance) is passed.
    """
    signing_key = key or load_or_create_signing_key()

    # Populate signer info BEFORE computing canonical bytes — the public
    # key is part of what gets signed (so a third party can't replay the
    # signature against a different claimed signer).
    public_key = signing_key.public_key()
    manifest.signer = SignerInfo(
        public_key=encode_public_key(public_key),
        key_id=fingerprint_public_key(public_key),
    )

    canonical = manifest.canonical_bytes()
    try:
        raw_signature = signing_key.sign(canonical)
    except Exception as exc:  # pragma: no cover - defensive
        raise SignatureError(f"Ed25519 signing failed: {exc!r}") from exc

    manifest.signature = "ed25519:" + base64.b64encode(raw_signature).decode("ascii")
    return manifest


__all__ = ["sign_manifest"]
