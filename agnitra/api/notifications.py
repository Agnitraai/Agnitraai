"""Notification helpers for hosted optimization jobs."""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


class WebhookNotifier:
    """Best-effort webhook sender used after async optimizations complete."""

    def __init__(self, *, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def notify(self, url: Optional[str], payload: Mapping[str, Any]) -> bool:
        if not url:
            return False
        try:
            import requests  # type: ignore
        except Exception:
            logger.warning("requests not available; skipping webhook notification to %s", url)
            return False

        try:
            response = requests.post(
                url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                logger.warning("Webhook %s returned status %s: %s", url, response.status_code, response.text)
                return False
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Failed to deliver webhook to %s: %s", url, exc)
            return False
        return True

