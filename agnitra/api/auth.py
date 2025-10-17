"""API key authentication helpers for the Agnitra cloud API."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence


def _normalise(values: Iterable[str]) -> Sequence[str]:
    return [value.strip() for value in values if value and value.strip()]


def _derive_digest(secret: str) -> bytes:
    """Return a digest used to compare API keys in constant time."""

    secret_bytes = secret.encode("utf-8")
    return hashlib.sha256(secret_bytes).digest()


@dataclass
class ApiKeyAuthenticator:
    """Simple API-key authenticator supporting hashed secrets."""

    accepted_digests: Sequence[bytes]

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, *, prefix: str = "AGNITRA_API_KEY") -> "ApiKeyAuthenticator":
        """Instantiate an authenticator from environment variables.

        Environment Variables
        ---------------------
        ``AGNITRA_API_KEY`` or ``AGNITRA_API_KEY_*`` entries contain either the
        raw API key or the hex-encoded SHA256 digest of the key. Raw keys are
        hashed in-memory before comparison.
        """

        env_map = env or os.environ
        candidates = []
        for key, value in env_map.items():
            if not key.startswith(prefix):
                continue
            if not value:
                continue
            candidates.append(value)
        digests: list[bytes] = []
        for candidate in _normalise(candidates):
            try:
                if len(candidate) == 64 and all(ch in "0123456789abcdef" for ch in candidate.lower()):
                    digests.append(bytes.fromhex(candidate))
                else:
                    digests.append(_derive_digest(candidate))
            except Exception:
                continue
        return cls(digests)

    def is_valid(self, api_key: Optional[str]) -> bool:
        """Return ``True`` when ``api_key`` matches a configured secret."""

        if not api_key:
            return False
        provided_digest = _derive_digest(api_key)
        for digest in self.accepted_digests:
            if len(digest) != len(provided_digest):
                continue
            if hmac.compare_digest(digest, provided_digest):
                return True
        return False

