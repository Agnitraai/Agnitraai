"""Buffered telemetry client for streaming usage events to the control plane."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


@dataclass
class TelemetryConfig:
    endpoint: Optional[str] = None
    batch_size: int = 50
    flush_interval: float = 3.0
    output_path: Optional[Path] = None
    request_timeout: float = 5.0
    api_key: Optional[str] = None


class TelemetryClient:
    """Buffers events and flushes them asynchronously."""

    def __init__(self, config: Optional[TelemetryConfig] = None) -> None:
        env_endpoint = os.getenv("AGNITRA_TELEMETRY_ENDPOINT")
        env_path = os.getenv("AGNITRA_TELEMETRY_PATH")
        cfg = config or TelemetryConfig()
        if env_endpoint:
            cfg.endpoint = env_endpoint
        if env_path and not cfg.output_path:
            cfg.output_path = Path(env_path)
        default_dir = Path("agnitraai/context")
        default_dir.mkdir(parents=True, exist_ok=True)
        if cfg.output_path is None:
            cfg.output_path = default_dir / "usage_events.jsonl"
        self.config = cfg

        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._last_flush = time.monotonic()
        self._flush_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._flush_inflight = False

    def emit(self, event: Mapping[str, Any]) -> None:
        enriched = dict(event)
        enriched.setdefault("event_id", str(uuid.uuid4()))
        enriched.setdefault("ts", int(time.time()))
        enriched.setdefault("emitted_at", time.time())
        enriched.setdefault("idempotency_key", enriched.get("event_id"))
        with self._lock:
            self._buffer.append(enriched)
            should_flush = len(self._buffer) >= self.config.batch_size or (
                time.monotonic() - self._last_flush >= self.config.flush_interval
            )
            if should_flush and not self._flush_inflight:
                self._schedule_flush()

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                self._last_flush = time.monotonic()
                return
            batch = list(self._buffer)
            self._buffer.clear()
            self._last_flush = time.monotonic()
            self._flush_inflight = True
        try:
            self._deliver(batch)
        finally:
            with self._lock:
                self._flush_inflight = False

    def close(self) -> None:
        self._stop_event.set()
        self.flush()
        thread = self._flush_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _schedule_flush(self) -> None:
        if self._flush_thread and self._flush_thread.is_alive():
            return
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        while not self._stop_event.is_set():
            time.sleep(self.config.flush_interval)
            self.flush()
            if not self._buffer:
                break

    def _deliver(self, batch: Iterable[Mapping[str, Any]]) -> None:
        if self.config.endpoint:
            self._post_batch(batch)
        else:
            self._write_local(batch)

    def _write_local(self, batch: Iterable[Mapping[str, Any]]) -> None:
        path = self.config.output_path
        if path is None:
            return
        lines = []
        for item in batch:
            try:
                lines.append(json.dumps(item, sort_keys=True))
            except TypeError:
                serialisable = json.loads(json.dumps(item, default=str))
                lines.append(json.dumps(serialisable, sort_keys=True))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                for line in lines:
                    fh.write(line + "\n")
        except Exception:
            pass

    def _post_batch(self, batch: Iterable[Mapping[str, Any]]) -> None:
        endpoint = self.config.endpoint
        if not endpoint:
            return
        payload = json.dumps({"events": list(batch)}).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.config.request_timeout) as resp:  # pragma: no cover
                resp.read()
        except Exception:
            # Fallback to local write so telemetry is not lost
            self._write_local(batch)


__all__ = ["TelemetryClient", "TelemetryConfig"]
