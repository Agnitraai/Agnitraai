"""Cloud marketplace billing adapters.

This module contains lightweight wrappers for the usage metering endpoints
exposed by the AWS, GCP, and Azure marketplaces. The adapters are designed to
operate in two modes:

* **Live** – When the relevant SDKs and environment variables are present the
  adapter submits a usage record to the cloud provider.
* **Deferred** – When credentials or optional dependencies are absent the
  adapter safely defers the submission and records the reason so that the
  caller can retry later.

The :class:`MarketplaceDispatcher` orchestrates multiple providers and returns a
sequence of :class:`BillingDispatchResult` objects describing the outcome for
each target marketplace.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from agnitra.core.metering.usage_meter import UsageEvent

LOGGER = logging.getLogger(__name__)


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(tz=_dt.timezone.utc)


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"1", "true", "yes", "y", "on"}:
            return True
        if normalised in {"0", "false", "no", "n", "off"}:
            return False
    return default


@dataclass
class MarketplaceUsageRecord:
    """Portable usage record sent to cloud marketplace endpoints."""

    project_id: str
    meter_name: str
    quantity: float
    currency: str = "USD"
    timestamp: _dt.datetime = field(default_factory=_utcnow)
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_event: Optional[UsageEvent] = None

    def to_external_payload(self) -> Dict[str, Any]:
        """Return a JSON-friendly payload shared with providers."""

        payload = {
            "project_id": self.project_id,
            "meter_name": self.meter_name,
            "quantity": self.quantity,
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }
        if self.model_name:
            payload["model_name"] = self.model_name
        return payload


@dataclass
class BillingDispatchResult:
    """Outcome of attempting to report a usage record to a provider."""

    provider: str
    status: str
    detail: str
    payload: Dict[str, Any] = field(default_factory=dict)


class BillingProvider:
    """Interface for marketplace billing adapters."""

    name: str = "provider"

    def report_usage(self, record: MarketplaceUsageRecord) -> BillingDispatchResult:  # pragma: no cover - interface
        raise NotImplementedError

    def pending_events(self) -> Sequence[MarketplaceUsageRecord]:
        """Return usage records buffered for retry."""

        return ()


class AWSMarketplaceProvider(BillingProvider):
    """Adapter for the AWS Marketplace Metering Service."""

    name = "aws"

    def __init__(
        self,
        *,
        product_code: Optional[str] = None,
        dimension: Optional[str] = None,
        region: Optional[str] = None,
        dry_run: Optional[bool] = None,
        client: Any = None,
    ) -> None:
        self.product_code = product_code or os.getenv("AWS_MARKETPLACE_PRODUCT_CODE")
        self.dimension = dimension or os.getenv("AWS_MARKETPLACE_USAGE_DIMENSION", "RUNTIME_OPTIMIZATION_HOURS")
        self.region = region or os.getenv("AWS_REGION")
        self.dry_run = _coerce_bool(
            dry_run if dry_run is not None else os.getenv("AWS_MARKETPLACE_DRY_RUN", "true"),
            default=True,
        )
        self._client = client or self._build_client()
        self._pending: List[MarketplaceUsageRecord] = []

    def _build_client(self) -> Any:
        try:  # pragma: no cover - optional dependency
            import boto3  # type: ignore
        except Exception:
            LOGGER.debug("boto3 is not available; AWS usage reports will be deferred.")
            return None

        try:
            return boto3.client("marketplace-metering", region_name=self.region)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Unable to create AWS Marketplace client: %s", exc)
            return None

    def report_usage(self, record: MarketplaceUsageRecord) -> BillingDispatchResult:
        if not self.product_code:
            detail = "AWS marketplace product code is not configured."
            LOGGER.warning(detail)
            return BillingDispatchResult(self.name, "skipped", detail, record.to_external_payload())

        request_payload = {
            "ProductCode": self.product_code,
            "Timestamp": record.timestamp,
            "UsageDimension": self.dimension,
            "UsageQuantity": max(int(round(record.quantity)), 1),
            "DryRun": self.dry_run,
        }

        if self._client is None:
            self._pending.append(record)
            return BillingDispatchResult(
                self.name,
                "deferred",
                "AWS marketplace client unavailable; stored usage for retry.",
                request_payload,
            )

        try:
            response = self._client.meter_usage(**request_payload)
            detail = json.dumps(response, default=str)
            return BillingDispatchResult(self.name, "ok", detail, request_payload)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to report AWS usage: %s", exc)
            self._pending.append(record)
            return BillingDispatchResult(self.name, "error", str(exc), request_payload)

    def pending_events(self) -> Sequence[MarketplaceUsageRecord]:
        return tuple(self._pending)


class GCPMarketplaceProvider(BillingProvider):
    """Adapter for the Google Cloud Commerce Metering API."""

    name = "gcp"

    def __init__(
        self,
        *,
        project_number: Optional[str] = None,
        service_account_file: Optional[str] = None,
        http_client: Any = None,
        api_endpoint: Optional[str] = None,
    ) -> None:
        self.project_number = project_number or os.getenv("GCP_PROJECT_NUMBER")
        self.service_account_file = service_account_file or os.getenv("GCP_MARKETPLACE_SERVICE_ACCOUNT")
        self.api_endpoint = api_endpoint or os.getenv(
            "GCP_MARKETPLACE_ENDPOINT",
            "https://cloudcommerceprocurement.googleapis.com/v1",
        )
        self._http_client = http_client
        self._pending: List[MarketplaceUsageRecord] = []
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[_dt.datetime] = None

    def report_usage(self, record: MarketplaceUsageRecord) -> BillingDispatchResult:
        payload = {
            "usage": {
                "startTime": record.timestamp.isoformat(),
                "endTime": record.timestamp.isoformat(),
                "amount": str(record.quantity),
                "currencyCode": record.currency,
                "usageReportingId": f"{record.project_id}:{record.meter_name}",
                "observedUsageUnits": str(record.quantity),
            },
            "metadata": dict(record.metadata),
        }

        if not self.project_number or not self.service_account_file:
            detail = "GCP marketplace credentials are not configured."
            LOGGER.warning(detail)
            return BillingDispatchResult(self.name, "skipped", detail, payload)

        client = self._http_client or _httpx_client()
        if client is None:
            self._pending.append(record)
            return BillingDispatchResult(
                self.name,
                "deferred",
                "httpx is unavailable; buffered GCP usage record.",
                payload,
            )

        access_token = self._get_access_token()
        if access_token is None:
            self._pending.append(record)
            return BillingDispatchResult(
                self.name,
                "error",
                "Unable to obtain Google Cloud marketplace access token.",
                payload,
            )

        service_name = record.metadata.get("gcp_service_name") or os.getenv("GCP_MARKETPLACE_SERVICE_NAME")
        sku_id = record.metadata.get("gcp_sku_id") or os.getenv("GCP_MARKETPLACE_SKU_ID")
        if not service_name or not sku_id:
            detail = "GCP marketplace service name or SKU id missing."
            LOGGER.warning(detail)
            return BillingDispatchResult(self.name, "skipped", detail, payload)

        url = f"{self.api_endpoint}/services/{service_name}/skus/{sku_id}:reportUsage"

        try:
            response = client.post(
                url,
                json=payload,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to report GCP usage: %s", exc)
            self._pending.append(record)
            return BillingDispatchResult(self.name, "error", str(exc), payload)

        if response.status_code >= 200 and response.status_code < 300:
            return BillingDispatchResult(self.name, "ok", response.text, payload)

        detail = f"GCP response {response.status_code}: {response.text}"
        LOGGER.error(detail)
        self._pending.append(record)
        return BillingDispatchResult(self.name, "error", detail, payload)

    def pending_events(self) -> Sequence[MarketplaceUsageRecord]:
        return tuple(self._pending)

    def _get_access_token(self) -> Optional[str]:
        """Return or refresh the Google Cloud access token."""

        if self.service_account_file is None:
            return None

        if (
            self._access_token
            and self._token_expiry
            and self._token_expiry - _utcnow() > _dt.timedelta(minutes=5)
        ):
            return self._access_token

        try:  # pragma: no cover - optional dependency
            from google.oauth2 import service_account  # type: ignore
            from google.auth.transport.requests import Request  # type: ignore
        except Exception:
            LOGGER.debug("google-auth is not available; cannot obtain GCP marketplace token.")
            return None

        try:
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
        except Exception as exc:
            LOGGER.exception("Failed to load GCP service account credentials: %s", exc)
            return None

        try:
            credentials.refresh(Request())
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to refresh GCP access token: %s", exc)
            return None

        token = getattr(credentials, "token", None)
        if isinstance(token, str) and token:
            self._access_token = token
            expiry = getattr(credentials, "expiry", None)
            if isinstance(expiry, _dt.datetime):
                self._token_expiry = expiry
            return token

        LOGGER.error("google-auth returned an empty marketplace access token.")
        return None


class AzureMarketplaceProvider(BillingProvider):
    """Adapter for the Azure Marketplace metering API."""

    name = "azure"

    def __init__(
        self,
        *,
        resource_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        tenant_id: Optional[str] = None,
        http_client: Any = None,
        token_endpoint: Optional[str] = None,
        metering_endpoint: Optional[str] = None,
    ) -> None:
        self.resource_id = resource_id or os.getenv("AZURE_MARKETPLACE_RESOURCE_ID")
        self.client_id = client_id or os.getenv("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("AZURE_CLIENT_SECRET")
        self.tenant_id = tenant_id or os.getenv("AZURE_TENANT_ID")
        self.token_endpoint = token_endpoint or os.getenv(
            "AZURE_TOKEN_ENDPOINT",
            f"https://login.microsoftonline.com/{self.tenant_id}/oauth2/token" if self.tenant_id else None,
        )
        self.metering_endpoint = metering_endpoint or os.getenv(
            "AZURE_MARKETPLACE_ENDPOINT",
            "https://marketplaceapi.microsoft.com/api/usageEvent?api-version=2017-03-31",
        )
        self._http_client = http_client
        self._cached_token: Optional[str] = None
        self._pending: List[MarketplaceUsageRecord] = []

    def report_usage(self, record: MarketplaceUsageRecord) -> BillingDispatchResult:
        if not self.resource_id:
            detail = "Azure marketplace resource id is not configured."
            LOGGER.warning(detail)
            return BillingDispatchResult(self.name, "skipped", detail, record.to_external_payload())

        client = self._http_client or _httpx_client()
        if client is None:
            self._pending.append(record)
            return BillingDispatchResult(
                self.name,
                "deferred",
                "httpx is unavailable; buffered Azure usage record.",
                record.to_external_payload(),
            )

        token = self._cached_token or self._fetch_token(client)
        if token is None:
            self._pending.append(record)
            return BillingDispatchResult(
                self.name,
                "error",
                "Azure marketplace access token is unavailable.",
                record.to_external_payload(),
            )

        payload = {
            "resourceId": self.resource_id,
            "quantity": record.quantity,
            "dimension": record.meter_name,
            "effectiveStartTime": record.timestamp.isoformat(),
            "planId": record.metadata.get("azure_plan_id") or os.getenv("AZURE_MARKETPLACE_PLAN_ID"),
            "resourceUsageId": record.metadata.get("azure_resource_usage_id")
            or f"{record.project_id}:{record.timestamp.isoformat()}",
            "properties": {"project_id": record.project_id, **record.metadata},
        }

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            response = client.post(self.metering_endpoint, json=payload, headers=headers)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to report Azure usage: %s", exc)
            self._pending.append(record)
            return BillingDispatchResult(self.name, "error", str(exc), payload)

        if response.status_code >= 200 and response.status_code < 300:
            return BillingDispatchResult(self.name, "ok", response.text, payload)

        detail = f"Azure response {response.status_code}: {response.text}"
        LOGGER.error(detail)
        self._pending.append(record)
        return BillingDispatchResult(self.name, "error", detail, payload)

    def _fetch_token(self, client: Any) -> Optional[str]:
        if not all([self.client_id, self.client_secret, self.tenant_id, self.token_endpoint]):
            LOGGER.warning("Azure marketplace credentials are incomplete.")
            return None

        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "resource": "https://management.core.windows.net/",
        }

        try:
            response = client.post(self.token_endpoint, data=data)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("Failed to obtain Azure token: %s", exc)
            return None

        if response.status_code >= 200 and response.status_code < 300:
            token = response.json().get("access_token")
            if isinstance(token, str):
                self._cached_token = token
                return token
        LOGGER.error("Azure token request failed with status %s", response.status_code)
        return None

    def pending_events(self) -> Sequence[MarketplaceUsageRecord]:
        return tuple(self._pending)


class MarketplaceDispatcher:
    """Coordinate usage dispatch across multiple marketplace providers."""

    def __init__(self, providers: Iterable[BillingProvider]) -> None:
        self._providers: Dict[str, BillingProvider] = {provider.name: provider for provider in providers}

    def provider_names(self) -> Sequence[str]:
        return tuple(self._providers.keys())

    def dispatch(
        self,
        record: MarketplaceUsageRecord,
        *,
        providers: Optional[Sequence[str]] = None,
    ) -> List[BillingDispatchResult]:
        targets = providers or self.provider_names()
        results: List[BillingDispatchResult] = []

        for provider_name in targets:
            provider = self._providers.get(provider_name)
            if provider is None:
                detail = f"Provider '{provider_name}' is not configured."
                LOGGER.warning(detail)
                results.append(BillingDispatchResult(provider_name, "skipped", detail, record.to_external_payload()))
                continue

            try:
                result = provider.report_usage(record)
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Provider %s raised during usage dispatch: %s", provider_name, exc)
                result = BillingDispatchResult(provider_name, "error", str(exc), record.to_external_payload())
            results.append(result)

        return results


def usage_event_to_record(
    event: UsageEvent,
    *,
    meter_name: str = "runtime_optimization_hours",
    quantity_field: str = "gpu_hours_after",
) -> MarketplaceUsageRecord:
    """Convert :class:`UsageEvent` into a marketplace usage record."""

    quantity_source = getattr(event, quantity_field, None)
    if quantity_source is None:
        quantity_source = event.total_billable

    metadata: Dict[str, Any] = dict(event.metadata)
    metadata.setdefault("baseline_latency_ms", event.baseline_latency_ms)
    metadata.setdefault("optimized_latency_ms", event.optimized_latency_ms)
    metadata.setdefault("tokens_processed", event.tokens_processed)

    return MarketplaceUsageRecord(
        project_id=event.project_id,
        meter_name=meter_name,
        quantity=_coerce_float(quantity_source, default=0.0),
        currency=event.currency,
        timestamp=event.timestamp,
        model_name=event.model_name,
        metadata=metadata,
        raw_event=event,
    )


def _httpx_client() -> Optional[Any]:
    try:  # pragma: no cover - optional dependency
        import httpx  # type: ignore
    except Exception:
        LOGGER.debug("httpx is not available for marketplace usage dispatch.")
        return None
    return httpx.Client(timeout=10.0)


def create_default_dispatcher() -> MarketplaceDispatcher:
    """Return dispatcher pre-populated with cloud marketplace adapters."""

    providers: List[BillingProvider] = [
        AWSMarketplaceProvider(),
        GCPMarketplaceProvider(),
        AzureMarketplaceProvider(),
    ]
    return MarketplaceDispatcher(providers)


__all__ = [
    "MarketplaceUsageRecord",
    "BillingDispatchResult",
    "BillingProvider",
    "AWSMarketplaceProvider",
    "GCPMarketplaceProvider",
    "AzureMarketplaceProvider",
    "MarketplaceDispatcher",
    "usage_event_to_record",
    "create_default_dispatcher",
]
