import asyncio
import hashlib

from agnitra.api.auth import ApiKeyAuthenticator
from agnitra.api.billing import StripeBillingClient
from agnitra.api.queue import OptimizationQueue


def test_api_key_authenticator_accepts_raw_and_hashed(monkeypatch):
    monkeypatch.setenv("AGNITRA_API_KEY_PRIMARY", "plain-secret")
    hashed = hashlib.sha256(b"hashed-secret").hexdigest()
    monkeypatch.setenv("AGNITRA_API_KEY_HASHED", hashed)

    authenticator = ApiKeyAuthenticator.from_env()

    assert authenticator.is_valid("plain-secret")
    assert authenticator.is_valid("hashed-secret")
    assert not authenticator.is_valid("invalid")


def test_stripe_billing_client_records_when_enabled(monkeypatch):
    monkeypatch.delenv("STRIPE_API_KEY", raising=False)
    client_disabled = StripeBillingClient.from_env()
    assert not client_disabled.enabled
    assert client_disabled.record_usage(customer_id="cust", quantity=1.0)["status"] == "disabled"

    monkeypatch.setenv("STRIPE_API_KEY", "sk_test_123")
    monkeypatch.setenv("STRIPE_METERED_PRICE_ID", "price_123")
    monkeypatch.setenv("STRIPE_ENABLED", "true")

    client = StripeBillingClient.from_env()
    result = client.record_usage(customer_id="cust-456", quantity=2.5, metadata={"project": "demo"})
    assert result["status"] == "recorded"
    assert client.records()[0]["customer_id"] == "cust-456"


def test_optimization_queue_processes_jobs():
    seen = {}

    async def _worker(payload):
        await asyncio.sleep(0.01)
        seen["value"] = payload["value"]
        return {"status": "ok", "value": payload["value"] * 2}

    async def _run():
        queue = OptimizationQueue(_worker, concurrency=2)
        job = await queue.enqueue({"value": 21})

        for _ in range(20):
            entry = queue.get(job.identifier)
            if entry and entry.status == "completed":
                break
            await asyncio.sleep(0.05)

        entry = queue.get(job.identifier)
        assert entry is not None
        assert entry.status == "completed"
        assert entry.result == {"status": "ok", "value": 42}

    asyncio.run(_run())
    assert seen["value"] == 21
