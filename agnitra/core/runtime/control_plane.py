"""Control plane client for retrieving optimization policies."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _post_json(url: str, data: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:  # pragma: no cover - network optional
            body = resp.read().decode("utf-8")
        return json.loads(body)
    except Exception:
        return None


@dataclass
class OptimizationPolicy:
    """Policy directives returned by the control plane."""

    policy_id: str = "default"
    plan_objective: str = "throughput"
    enable_llm: bool = True
    enable_rl: bool = True
    calibration_iterations: int = 40
    calibration_warmup: int = 5
    cache_ttl_seconds: int = 86400
    telemetry_sample_rate: float = 1.0
    default_preset: Optional[Dict[str, Any]] = None
    auto_retrain: bool = False
    llm_model: Optional[str] = None
    pass_presets: Optional[List[Dict[str, Any]]] = None
    abtest_iterations: Optional[int] = None
    abtest_warmup: Optional[int] = None
    auto_retrain_interval: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "policy_id": self.policy_id,
            "plan_objective": self.plan_objective,
            "enable_llm": self.enable_llm,
            "enable_rl": self.enable_rl,
            "calibration_iterations": self.calibration_iterations,
            "calibration_warmup": self.calibration_warmup,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "telemetry_sample_rate": self.telemetry_sample_rate,
            "default_preset": self.default_preset,
            "auto_retrain": self.auto_retrain,
            "llm_model": self.llm_model,
            "pass_presets": self.pass_presets,
            "abtest_iterations": self.abtest_iterations,
            "abtest_warmup": self.abtest_warmup,
            "auto_retrain_interval": self.auto_retrain_interval,
        }
        payload.update(self.extra)
        return payload


class ControlPlaneClient:
    """Fetch optimization policies with caching and graceful fallbacks."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        policy_path: Optional[Path] = None,
        default_policy: Optional[OptimizationPolicy] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("AGNITRA_CONTROL_PLANE_URL")
        policy_file_env = os.getenv("AGNITRA_POLICY_PATH")
        self.policy_path = policy_path or (Path(policy_file_env) if policy_file_env else None)
        self._lock = threading.Lock()
        self._cache: Dict[str, tuple[float, OptimizationPolicy]] = {}
        self._default = default_policy or OptimizationPolicy()

    def fetch_policy(self, project_id: str, fingerprint: Mapping[str, Any]) -> OptimizationPolicy:
        cache_key = project_id or "default"
        now = time.time()

        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and (now - cached[0]) < 30.0:
                return cached[1]

        policy = self._load_policy_from_file(project_id, fingerprint)
        if policy is None:
            policy = self._load_policy_from_http(project_id, fingerprint)

        if policy is None:
            policy = self._default

        with self._lock:
            self._cache[cache_key] = (now, policy)
        return policy

    def _load_policy_from_file(
        self,
        project_id: str,
        fingerprint: Mapping[str, Any],
    ) -> Optional[OptimizationPolicy]:
        if not self.policy_path:
            return None
        payload = _read_json(self.policy_path)
        if not payload:
            return None
        return self._policy_from_payload(payload, project_id, fingerprint)

    def _load_policy_from_http(
        self,
        project_id: str,
        fingerprint: Mapping[str, Any],
    ) -> Optional[OptimizationPolicy]:
        if not self.base_url:
            return None
        payload = {
            "project_id": project_id,
            "fingerprint": fingerprint,
        }
        response = _post_json(self.base_url.rstrip("/") + "/v1/policy/fetch", payload)
        if not response:
            return None
        return self._policy_from_payload(response, project_id, fingerprint)

    def _policy_from_payload(
        self,
        payload: Mapping[str, Any],
        project_id: str,
        fingerprint: Mapping[str, Any],
    ) -> OptimizationPolicy:
        extra = dict(payload.get("extra", {}))
        if "project_id" not in extra:
            extra["project_id"] = project_id
        extra["fingerprint_signature"] = payload.get("fingerprint_signature")
        pass_presets = payload.get("pass_presets") or payload.get("passes")
        if isinstance(pass_presets, list):
            extra["pass_presets"] = pass_presets
        llm_model = payload.get("llm_model")
        if llm_model:
            extra["llm_model"] = llm_model
        return OptimizationPolicy(
            policy_id=str(payload.get("policy_id", "default")),
            plan_objective=str(payload.get("plan", {}).get("objective", payload.get("plan_objective", "throughput"))),
            enable_llm=bool(payload.get("enable_llm", True)),
            enable_rl=bool(payload.get("enable_rl", True)),
            calibration_iterations=int(payload.get("calibration_iterations", payload.get("repeats", 40))),
            calibration_warmup=int(payload.get("calibration_warmup", payload.get("warmup", 5))),
            cache_ttl_seconds=int(payload.get("cache_ttl_seconds", 86400)),
            telemetry_sample_rate=float(payload.get("telemetry_sample_rate", 1.0)),
            default_preset=payload.get("default_preset") or payload.get("preset"),
            auto_retrain=bool(payload.get("auto_retrain", False)),
            llm_model=payload.get("llm_model"),
            pass_presets=pass_presets if isinstance(pass_presets, list) else None,
            abtest_iterations=payload.get("abtest_iterations"),
            abtest_warmup=payload.get("abtest_warmup"),
            auto_retrain_interval=payload.get("auto_retrain_interval"),
            extra=extra,
        )


__all__ = ["ControlPlaneClient", "OptimizationPolicy"]
