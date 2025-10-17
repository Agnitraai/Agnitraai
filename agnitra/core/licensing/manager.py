"""Enterprise SDK license management and validation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Set

from .gpu_tracker import GpuUsageTracker, collect_gpu_fingerprints


class LicenseValidationError(RuntimeError):
    """Raised when a license file is missing or invalid."""


def _default_license_path() -> Path:
    base = os.environ.get("AGNITRA_LICENSE_PATH")
    if base:
        return Path(base).expanduser()
    return Path.home() / ".agnitra" / "license.json"


def _ensure_directory(path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - defensive
        raise LicenseValidationError(f"Unable to create license directory {path.parent}: {exc}") from exc


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    raise LicenseValidationError("License expiry could not be parsed.")


def _normalise_feature(value: str) -> str:
    return value.strip().lower()


def _compute_signature(payload: Mapping[str, Any], secret: str) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
    return digest


@dataclass
class LicenseRecord:
    """Parsed representation of an enterprise SDK license."""

    license_key: str
    customer: str
    expires_at: datetime
    seats: int
    features: Sequence[str]
    signature: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_expired(self, *, grace_period_days: int = 0) -> bool:
        if grace_period_days <= 0:
            return datetime.now(timezone.utc) > self.expires_at
        grace_delta = datetime.now(timezone.utc) - self.expires_at
        return grace_delta.total_seconds() > grace_period_days * 86400

    def feature_set(self) -> Set[str]:
        return {_normalise_feature(feature) for feature in self.features}


class LicenseManager:
    """High-level helper for validating enterprise SDK licenses."""

    def __init__(
        self,
        *,
        secret: Optional[str] = None,
        storage_path: Optional[Path] = None,
        gpu_tracker: Optional[GpuUsageTracker] = None,
    ) -> None:
        self.secret = secret or os.environ.get("AGNITRA_LICENSE_SECRET")
        self.storage_path = Path(storage_path) if storage_path else _default_license_path()
        self.gpu_tracker = gpu_tracker or GpuUsageTracker()
        self._license: Optional[LicenseRecord] = None
        self._active_seats: Set[str] = set()
        self._loaded_payload: Optional[Mapping[str, Any]] = None

    # ------------------------------------------------------------------
    # Loading & validation
    # ------------------------------------------------------------------
    def load(self) -> LicenseRecord:
        if self._license is not None and not self._license.is_expired():
            return self._license

        if not self.storage_path.exists():
            raise LicenseValidationError(
                f"License file not found at {self.storage_path}. "
                "Provide a valid license via AGNITRA_LICENSE_PATH."
            )

        try:
            raw = self.storage_path.read_text(encoding="utf-8")
        except Exception as exc:
            raise LicenseValidationError(f"Unable to read license file: {exc}") from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LicenseValidationError(f"License file contains invalid JSON: {exc}") from exc

        if not isinstance(payload, MutableMapping):
            raise LicenseValidationError("License payload must be a JSON object.")

        try:
            record = LicenseRecord(
                license_key=str(payload["license_key"]),
                customer=str(payload["customer"]),
                expires_at=_parse_datetime(payload["expires_at"]),
                seats=int(payload.get("seats", 1)),
                features=list(payload.get("features", [])),
                signature=payload.get("signature"),
                metadata=payload.get("metadata", {}),
            )
        except KeyError as exc:
            raise LicenseValidationError(f"Missing license field: {exc}") from exc
        except Exception as exc:
            raise LicenseValidationError(f"Unable to parse license: {exc}") from exc

        if record.is_expired():
            raise LicenseValidationError("License expired.")

        if self.secret:
            expected_sig = _compute_signature(_canonical_payload(payload), self.secret)
            if not record.signature or not hmac.compare_digest(expected_sig, record.signature):
                raise LicenseValidationError("License signature verification failed.")

        self._license = record
        self._loaded_payload = payload
        return record

    def save(self, payload: Mapping[str, Any]) -> LicenseRecord:
        _ensure_directory(self.storage_path)
        if self.secret and "signature" not in payload:
            payload = dict(payload)
            payload["signature"] = _compute_signature(_canonical_payload(payload), self.secret)
        self.storage_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._license = None
        return self.load()

    # ------------------------------------------------------------------
    # Feature gating & seat tracking
    # ------------------------------------------------------------------
    def ensure_feature(self, feature: str) -> None:
        record = self.load()
        feature_normalised = _normalise_feature(feature)
        if feature_normalised not in record.feature_set():
            raise LicenseValidationError(f"License missing required feature: {feature}.")

    def checkout_seat(self, seat_id: str) -> None:
        record = self.load()
        self._active_seats.add(seat_id)
        if len(self._active_seats) > record.seats:
            raise LicenseValidationError("Seat limit exceeded for this license.")

    def release_seat(self, seat_id: str) -> None:
        self._active_seats.discard(seat_id)

    # ------------------------------------------------------------------
    # GPU usage monitoring (per-GPU licensing)
    # ------------------------------------------------------------------
    def register_gpu_run(self, *, org_id: Optional[str] = None) -> Dict[str, Any]:
        record = self.load()
        fingerprints = collect_gpu_fingerprints()
        usage_record = self.gpu_tracker.register_run(
            license_key=record.license_key,
            org_id=org_id or record.customer,
            fingerprints=fingerprints,
        )
        gpu_limit = int(_safe_int(self._loaded_payload.get("gpu_limit"))) if self._loaded_payload else None
        if gpu_limit and usage_record["active_gpu_count"] > gpu_limit:
            raise LicenseValidationError("Per-GPU license limit exceeded.")
        return usage_record

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        record = self.load()
        payload = {
            "license_key": record.license_key,
            "customer": record.customer,
            "expires_at": record.expires_at.isoformat(),
            "seats": record.seats,
            "features": list(record.features),
        }
        if record.signature:
            payload["signature"] = record.signature
        payload["metadata"] = dict(record.metadata)
        return payload


def _canonical_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    retain_keys = {"license_key", "customer", "expires_at", "seats", "features", "metadata"}
    return {key: payload[key] for key in retain_keys if key in payload}


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0

