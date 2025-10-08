"""Local optimization cache for storing winning profiles."""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class CachedProfile:
    signature: str
    created_at: float
    ttl_seconds: int
    payload: Dict[str, Any]

    def is_expired(self, now: Optional[float] = None) -> bool:
        current = now or time.time()
        return (current - self.created_at) > max(self.ttl_seconds, 0)


class OptimizationCache:
    """Simple JSON-backed cache keyed by fingerprint signatures."""

    def __init__(self, *, path: Optional[Path] = None) -> None:
        default_dir = Path(os.getenv("AGNITRA_CACHE_DIR", "agnitraai/context"))
        default_dir.mkdir(parents=True, exist_ok=True)
        self.path = path or (default_dir / "optimization_cache.json")
        self._lock = threading.Lock()
        self._data: Dict[str, CachedProfile] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.path.exists():
            self._loaded = True
            return
        try:
            raw = json.loads(self.path.read_text())
        except Exception:
            self._loaded = True
            return
        for key, entry in raw.items():
            if not isinstance(entry, dict):
                continue
            profile = CachedProfile(
                signature=str(entry.get("signature", key)),
                created_at=float(entry.get("created_at", 0.0)),
                ttl_seconds=int(entry.get("ttl_seconds", 0)),
                payload=dict(entry.get("payload", {})),
            )
            self._data[key] = profile
        self._loaded = True

    def lookup(self, signature: str) -> Optional[CachedProfile]:
        with self._lock:
            self._ensure_loaded()
            profile = self._data.get(signature)
            if profile and profile.is_expired():
                self._data.pop(signature, None)
                self._persist()
                return None
            return profile

    def store(
        self,
        signature: str,
        payload: Mapping[str, Any],
        *,
        ttl_seconds: int,
    ) -> CachedProfile:
        profile = CachedProfile(
            signature=signature,
            created_at=time.time(),
            ttl_seconds=ttl_seconds,
            payload=dict(payload),
        )
        with self._lock:
            self._ensure_loaded()
            self._data[signature] = profile
            self._persist()
        return profile

    def clear_expired(self) -> None:
        with self._lock:
            self._ensure_loaded()
            now = time.time()
            expired = [key for key, profile in self._data.items() if profile.is_expired(now)]
            for key in expired:
                self._data.pop(key, None)
            if expired:
                self._persist()

    def _persist(self) -> None:
        serialisable = {
            key: {
                "signature": profile.signature,
                "created_at": profile.created_at,
                "ttl_seconds": profile.ttl_seconds,
                "payload": profile.payload,
            }
            for key, profile in self._data.items()
        }
        try:
            self.path.write_text(json.dumps(serialisable, indent=2, sort_keys=True))
        except Exception:
            pass


__all__ = ["OptimizationCache", "CachedProfile"]

