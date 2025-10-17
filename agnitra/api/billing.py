"""Stripe billing integration helpers for the hosted SaaS offering."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


@dataclass
class StripeBillingClient:
    """Minimal Stripe usage-record client.

    Production deployments should replace this shim with the official Stripe
    SDK.  The shim keeps the hosted worker / API logic decoupled from Stripe so
    that unit tests and offline development can run without network access.
    """

    api_key: Optional[str] = None
    metered_price_id: Optional[str] = None
    enabled: bool = field(default=False)
    _records: list[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "StripeBillingClient":
        env_map = env or os.environ
        api_key = env_map.get("STRIPE_API_KEY")
        price_id = env_map.get("STRIPE_METERED_PRICE_ID")
        enabled = _coerce_bool(env_map.get("STRIPE_ENABLED", api_key is not None and price_id is not None))
        return cls(api_key=api_key, metered_price_id=price_id, enabled=enabled)

    def record_usage(self, *, customer_id: str, quantity: float, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Record a usage event for ``customer_id``.

        In the OSS version we simply log the payload. Hosted deployments can
        override this class or monkeypatch ``record_usage`` to call Stripe.
        """

        if not self.enabled:
            logger.debug("Stripe billing disabled; ignoring usage event for customer %s", customer_id)
            return {"status": "disabled"}

        payload = {
            "customer_id": customer_id,
            "quantity": quantity,
            "price_id": self.metered_price_id,
            "metadata": dict(metadata or {}),
        }
        self._records.append(payload)
        logger.info("Recorded Stripe usage: %s", json.dumps(payload, sort_keys=True))
        return {"status": "recorded", "payload": payload}

    def records(self) -> list[Dict[str, Any]]:
        """Return a snapshot of usage events registered via this client."""

        return list(self._records)

