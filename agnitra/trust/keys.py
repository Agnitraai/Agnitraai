"""Ed25519 signing key management.

Resolution order for the signing key (first match wins):

1. ``AGNITRA_TRUST_KEY_PEM`` env var — PEM-encoded private key, inline.
   Useful for CI / containers where you don't want a file on disk.
2. ``AGNITRA_TRUST_KEY_PATH`` env var — path to a PEM-encoded private key.
3. ``~/.agnitra/keys/signing.pem`` — the default location.
4. Generate a fresh ephemeral keypair, persist to (3), warn loudly.

Why Ed25519: small (32-byte public keys, 64-byte signatures), fast
(microseconds per sign/verify), strong security margin, no nonce-reuse
footguns. The Python ``cryptography`` package wraps libsodium-compatible
implementations.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from agnitra.trust.errors import CryptographyMissingError, SignatureError

LOGGER = logging.getLogger(__name__)

DEFAULT_KEY_DIR = Path.home() / ".agnitra" / "keys"
DEFAULT_KEY_FILE = DEFAULT_KEY_DIR / "signing.pem"


def _import_cryptography():
    """Import ``cryptography`` lazily; surface a clear error if missing."""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
        return serialization, Ed25519PrivateKey, Ed25519PublicKey
    except ImportError as exc:
        raise CryptographyMissingError(
            "The trust layer requires `cryptography>=42.0`. "
            "Install with: pip install \"agnitra[trust]\""
        ) from exc


def _generate_keypair():
    _, Ed25519PrivateKey, _ = _import_cryptography()
    return Ed25519PrivateKey.generate()


def _serialize_private_key_pem(key) -> bytes:
    serialization, _, _ = _import_cryptography()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _load_private_key_pem(pem_bytes: bytes):
    serialization, Ed25519PrivateKey, _ = _import_cryptography()
    try:
        key = serialization.load_pem_private_key(pem_bytes, password=None)
    except Exception as exc:
        raise SignatureError(f"Failed to parse PEM private key: {exc!r}") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise SignatureError(
            "Loaded key is not Ed25519. The trust layer only supports "
            "Ed25519 keys; convert or regenerate before signing."
        )
    return key


def fingerprint_public_key(public_key) -> str:
    """Stable short ID — first 16 hex chars of SHA-256(raw public bytes).

    Used as the human-readable ``key_id`` in signed manifests so callers
    can identify the signer at a glance without parsing the full
    base64 public key.
    """
    raw = public_key.public_bytes_raw()
    return hashlib.sha256(raw).hexdigest()[:16]


def encode_public_key(public_key) -> str:
    """``"ed25519:<base64>"`` form for inclusion in manifests."""
    import base64
    return "ed25519:" + base64.b64encode(public_key.public_bytes_raw()).decode("ascii")


def decode_public_key(encoded: str):
    """Inverse of :func:`encode_public_key`. Raises :class:`SignatureError`."""
    import base64
    _, _, Ed25519PublicKey = _import_cryptography()
    if not encoded.startswith("ed25519:"):
        raise SignatureError(f"Public key has unsupported algorithm prefix: {encoded[:16]!r}")
    raw = base64.b64decode(encoded.removeprefix("ed25519:"))
    if len(raw) != 32:
        raise SignatureError(
            f"Ed25519 public keys are 32 bytes; got {len(raw)} after base64 decode"
        )
    return Ed25519PublicKey.from_public_bytes(raw)


def load_or_create_signing_key(*, key_path: Optional[Path] = None):
    """Resolve the signing key per the module docstring's order.

    ``key_path`` overrides the default file location for callers (like
    the CLI's ``trust keys generate``) that want explicit control.
    """
    pem_inline = os.environ.get("AGNITRA_TRUST_KEY_PEM")
    if pem_inline:
        return _load_private_key_pem(pem_inline.encode("utf-8"))

    env_path = os.environ.get("AGNITRA_TRUST_KEY_PATH")
    if env_path:
        path = Path(env_path)
        if not path.exists():
            raise SignatureError(
                f"AGNITRA_TRUST_KEY_PATH points at {path} but no file exists there."
            )
        return _load_private_key_pem(path.read_bytes())

    target = key_path or DEFAULT_KEY_FILE
    if target.exists():
        return _load_private_key_pem(target.read_bytes())

    LOGGER.warning(
        "No trust signing key found at %s. Generating an ephemeral keypair "
        "and persisting it. This is OK for local development; for production "
        "set AGNITRA_TRUST_KEY_PATH to a key managed by your secret store.",
        target,
    )
    key = _generate_keypair()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_serialize_private_key_pem(key))
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - non-POSIX
        pass
    return key


def generate_keypair_pem(*, key_path: Optional[Path] = None) -> Tuple[Path, str]:
    """Create a new keypair and persist the private half. Returns ``(path, key_id)``."""
    target = key_path or DEFAULT_KEY_FILE
    if target.exists():
        raise SignatureError(
            f"Refusing to overwrite existing key at {target}. "
            "Move or delete it first."
        )
    key = _generate_keypair()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_serialize_private_key_pem(key))
    try:
        os.chmod(target, 0o600)
    except OSError:  # pragma: no cover - non-POSIX
        pass
    return target, fingerprint_public_key(key.public_key())


__all__ = [
    "DEFAULT_KEY_DIR",
    "DEFAULT_KEY_FILE",
    "decode_public_key",
    "encode_public_key",
    "fingerprint_public_key",
    "generate_keypair_pem",
    "load_or_create_signing_key",
]
