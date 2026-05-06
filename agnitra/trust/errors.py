"""Trust-layer errors.

All trust errors derive from ``TrustError`` so callers can catch the
whole class. Verification failures are NOT errors — they are returned
as :class:`agnitra.trust.VerifyResult` instances. Exceptions here are
for misuse (bad input, missing crypto deps), not for "the signature
didn't match."
"""
from __future__ import annotations


class TrustError(Exception):
    """Base class for every trust-layer error."""


class ManifestError(TrustError):
    """Manifest is structurally invalid (missing fields, wrong types)."""


class SignatureError(TrustError):
    """Cryptography failure: missing key, malformed signature, bad encoding."""


class CryptographyMissingError(TrustError):
    """The ``cryptography`` package is not installed.

    Raised at the point of first crypto operation so callers can detect
    "trust layer unavailable" without having to import-time-guard.
    Install with: ``pip install "agnitra[trust]"``.
    """


__all__ = [
    "CryptographyMissingError",
    "ManifestError",
    "SignatureError",
    "TrustError",
]
