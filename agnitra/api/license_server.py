"""Lightweight Starlette application acting as the license control plane."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from agnitra.core.licensing import LicenseValidationError


@dataclass
class LicenseStore:
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def register(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        key = payload.get("license_key")
        if not key:
            raise LicenseValidationError("license_key is required")
        record = self.records.setdefault(key, {"usage": []})
        record.setdefault("customer", payload.get("customer"))
        record.setdefault("features", payload.get("features", []))
        record.setdefault("seats", payload.get("seats", 1))
        record.setdefault("created_at", datetime.now(timezone.utc).isoformat())
        usage = payload.get("usage")
        if isinstance(usage, dict):
            record["usage"].append(usage)
        return record


_LICENSE_STORE = LicenseStore()


async def register_license(request: Request) -> Response:
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        return JSONResponse({"error": f"Invalid JSON: {exc}"}, status_code=400)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "Payload must be a JSON object"}, status_code=400)
    try:
        record = _LICENSE_STORE.register(payload)
    except LicenseValidationError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"status": "ok", "record": record})


async def health(_: Request) -> Response:
    return JSONResponse({"status": "ok"})


def create_license_app() -> Starlette:
    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/register", register_license, methods=["POST"]),
    ]
    return Starlette(debug=False, routes=routes)
