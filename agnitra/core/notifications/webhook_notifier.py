"""Webhook notifier for optimization results.

Supports Slack incoming webhooks, Discord webhooks, and Telegram Bot API so
optimization results can be delivered to the developer's preferred channel —
mirroring OpenClaw's multi-channel notification approach.

Configure via environment variables::

    AGNITRA_NOTIFY_WEBHOOK_URL=https://hooks.slack.com/services/...
    AGNITRA_NOTIFY_CHANNEL=slack   # slack | discord | telegram

Or pass arguments directly to :class:`WebhookNotifier`.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from agnitra.core.runtime.agent import RuntimeOptimizationResult

LOGGER = logging.getLogger(__name__)

_SUPPORTED_CHANNELS = {"slack", "discord", "telegram", "generic"}


class WebhookNotifier:
    """Post optimization result summaries to Slack, Discord, Telegram, or any webhook URL.

    Parameters
    ----------
    url:
        Webhook URL. Falls back to ``AGNITRA_NOTIFY_WEBHOOK_URL`` env var.
    channel:
        Target channel format: ``"slack"``, ``"discord"``, ``"telegram"``, or
        ``"generic"`` (raw JSON). Falls back to ``AGNITRA_NOTIFY_CHANNEL`` env var.
    telegram_chat_id:
        Required when ``channel="telegram"``. The chat/group ID to send to.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: Optional[str] = None,
        channel: str = "slack",
        *,
        telegram_chat_id: Optional[str] = None,
        timeout: float = 5.0,
    ) -> None:
        self.url = url or os.environ.get("AGNITRA_NOTIFY_WEBHOOK_URL", "")
        self.channel = (
            os.environ.get("AGNITRA_NOTIFY_CHANNEL", channel).strip().lower()
        )
        self.telegram_chat_id = telegram_chat_id or os.environ.get(
            "AGNITRA_NOTIFY_TELEGRAM_CHAT_ID", ""
        )
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "WebhookNotifier":
        """Build a notifier from environment variables."""
        return cls(
            url=os.environ.get("AGNITRA_NOTIFY_WEBHOOK_URL"),
            channel=os.environ.get("AGNITRA_NOTIFY_CHANNEL", "slack"),
            telegram_chat_id=os.environ.get("AGNITRA_NOTIFY_TELEGRAM_CHAT_ID"),
        )

    def is_configured(self) -> bool:
        """Return True when a webhook URL is set."""
        return bool(self.url)

    def notify(self, result: "RuntimeOptimizationResult") -> bool:
        """Send a summary of *result* to the configured webhook.

        Returns True on success, False on any failure (errors are logged but
        never raised so the optimization caller is not disrupted).
        """
        if not self.url:
            return False

        try:
            payload = self._build_payload(result)
            return self._post(self.url, payload)
        except Exception as exc:
            LOGGER.warning("WebhookNotifier failed to send notification: %s", exc)
            return False

    def notify_raw(self, url: str, payload: Mapping[str, Any]) -> bool:
        """Post an arbitrary *payload* dict to *url* (best-effort)."""
        try:
            return self._post(url, dict(payload))
        except Exception as exc:
            LOGGER.warning("WebhookNotifier.notify_raw failed: %s", exc)
            return False

    def _build_payload(self, result: "RuntimeOptimizationResult") -> Dict[str, Any]:
        """Convert result into the channel-appropriate payload format."""

        baseline_ms = result.baseline.latency_ms
        optimized_ms = result.optimized.latency_ms
        improvement_ms = baseline_ms - optimized_ms
        improvement_pct = (improvement_ms / baseline_ms * 100.0) if baseline_ms else 0.0

        usage = result.usage_event
        project_id = result.notes.get("project_id", "unknown") if isinstance(result.notes, dict) else "unknown"
        model_name = result.notes.get("model_name", "model") if isinstance(result.notes, dict) else "model"

        summary_lines = [
            f"*Agnitra optimization complete* — `{model_name}` (project: `{project_id}`)",
            f"• Baseline: `{baseline_ms:.2f} ms` → Optimized: `{optimized_ms:.2f} ms`"
            f" ({improvement_pct:+.1f}%)",
        ]
        if usage is not None:
            summary_lines.append(
                f"• GPU hours saved: `{usage.gpu_hours_saved:.6f}` "
                f"| Billable: `{usage.total_billable:.4f} {usage.currency}`"
            )

        summary = "\n".join(summary_lines)

        if self.channel == "slack":
            return {"text": summary}

        if self.channel == "discord":
            return {"content": summary.replace("*", "**").replace("`", "`")}

        if self.channel == "telegram":
            return {
                "chat_id": self.telegram_chat_id,
                "text": summary.replace("*", "").replace("`", ""),
                "parse_mode": "Markdown",
            }

        return {
            "project_id": project_id,
            "model_name": model_name,
            "baseline_latency_ms": baseline_ms,
            "optimized_latency_ms": optimized_ms,
            "improvement_pct": improvement_pct,
            "gpu_hours_saved": usage.gpu_hours_saved if usage else None,
            "total_billable": usage.total_billable if usage else None,
            "currency": usage.currency if usage else "USD",
        }

    def _post(self, url: str, payload: Dict[str, Any]) -> bool:
        try:
            import httpx  # type: ignore[import-not-found]
            response = httpx.post(
                url,
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            if response.status_code >= 400:
                LOGGER.warning(
                    "Webhook %s returned %s: %s", url, response.status_code, response.text[:200]
                )
                return False
            return True
        except ImportError:
            pass

        try:
            import urllib.request
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=int(self.timeout)) as resp:
                return resp.status < 400
        except Exception as exc:
            LOGGER.warning("Webhook POST to %s failed: %s", url, exc)
            return False


__all__ = ["WebhookNotifier"]
