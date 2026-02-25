"""Background heartbeat scheduler for periodic re-optimization.

Mirrors OpenClaw's heartbeat system: a daemon thread wakes every N seconds,
reads pending jobs from ``agnitraai/context/auto_retrain_jobs.jsonl``, and
triggers re-optimization for any job whose interval has elapsed.

Usage::

    from agnitra.core.runtime.heartbeat import OptimizationHeartbeat

    hb = OptimizationHeartbeat(interval_seconds=1800)
    hb.start()   # non-blocking, runs in daemon thread
    ...
    hb.stop()

Or from the CLI::

    agnitra heartbeat --interval 30
    agnitra heartbeat --once
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOGGER = logging.getLogger(__name__)

_DEFAULT_JOBS_PATH = Path("agnitraai/context/auto_retrain_jobs.jsonl")
_DEFAULT_RESULTS_PATH = Path("agnitraai/context/heartbeat_results.jsonl")


class OptimizationHeartbeat:
    """Daemon thread that periodically re-optimizes tracked model workloads.

    Parameters
    ----------
    interval_seconds:
        Seconds between heartbeat cycles (default: 1800 = 30 minutes).
    jobs_path:
        Path to the JSONL file written by :meth:`RuntimeOptimizationAgent._schedule_auto_retrain`.
    results_path:
        Path where heartbeat run results are appended.
    on_cycle:
        Optional callback invoked with the list of processed job records after
        each cycle. Useful for testing or custom integrations.
    """

    def __init__(
        self,
        interval_seconds: int = 1800,
        *,
        jobs_path: Optional[Path] = None,
        results_path: Optional[Path] = None,
        on_cycle: Optional[Callable[[List[Dict[str, Any]]], None]] = None,
    ) -> None:
        self.interval_seconds = max(1, int(interval_seconds))
        self.jobs_path = jobs_path or _DEFAULT_JOBS_PATH
        self.results_path = results_path or _DEFAULT_RESULTS_PATH
        self.on_cycle = on_cycle
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cycle_count = 0

    def start(self) -> None:
        """Start the heartbeat in a background daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            LOGGER.debug("Heartbeat already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="agnitra-heartbeat",
            daemon=True,
        )
        self._thread.start()
        LOGGER.info(
            "Heartbeat started (interval=%ds, jobs_path=%s).",
            self.interval_seconds,
            self.jobs_path,
        )

    def stop(self) -> None:
        """Signal the heartbeat thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        LOGGER.info("Heartbeat stopped after %d cycle(s).", self._cycle_count)

    def run_once(self) -> List[Dict[str, Any]]:
        """Execute a single heartbeat cycle and return processed job records."""
        return self._process_jobs()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                processed = self._process_jobs()
                self._cycle_count += 1
                if self.on_cycle:
                    try:
                        self.on_cycle(processed)
                    except Exception:
                        LOGGER.debug("on_cycle callback raised", exc_info=True)
            except Exception:
                LOGGER.exception("Heartbeat cycle failed")
            self._stop_event.wait(timeout=self.interval_seconds)

    def _process_jobs(self) -> List[Dict[str, Any]]:
        jobs = self._read_pending_jobs()
        if not jobs:
            LOGGER.debug("Heartbeat: no pending jobs found.")
            return []

        now = time.time()
        processed: List[Dict[str, Any]] = []

        for job in jobs:
            result = self._run_job(job, now)
            if result is not None:
                processed.append(result)

        if processed:
            self._append_results(processed)
            LOGGER.info("Heartbeat: processed %d job(s).", len(processed))

        return processed

    def _read_pending_jobs(self) -> List[Dict[str, Any]]:
        if not self.jobs_path.exists():
            return []
        jobs: List[Dict[str, Any]] = []
        try:
            with self.jobs_path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        jobs.append(json.loads(line))
                    except json.JSONDecodeError:
                        LOGGER.debug("Heartbeat: skipping malformed job line.")
        except OSError as exc:
            LOGGER.warning("Heartbeat: could not read jobs file: %s", exc)
        return jobs

    def _run_job(self, job: Dict[str, Any], now: float) -> Optional[Dict[str, Any]]:
        created_at = job.get("created_at", 0.0)
        retrain_req = job.get("retrain_request", {})
        auto_retrain_interval = job.get("auto_retrain_interval")

        if auto_retrain_interval is not None:
            try:
                next_run_at = float(created_at) + float(auto_retrain_interval)
                if now < next_run_at:
                    LOGGER.debug(
                        "Heartbeat: job for policy %s not yet due (%.0fs remaining).",
                        job.get("policy_id"),
                        next_run_at - now,
                    )
                    return None
            except (TypeError, ValueError):
                pass

        project_id = job.get("project_id", "default")
        model_name = job.get("model_name", "unknown")
        policy_id = job.get("policy_id")
        fingerprint_sig = job.get("fingerprint_signature")

        LOGGER.info(
            "Heartbeat: triggering re-optimization for project=%s model=%s policy=%s",
            project_id,
            model_name,
            policy_id,
        )

        result_record: Dict[str, Any] = {
            "heartbeat_ts": now,
            "project_id": project_id,
            "model_name": model_name,
            "policy_id": policy_id,
            "fingerprint_signature": fingerprint_sig,
            "retrain_request": retrain_req,
            "status": "dispatched",
        }

        try:
            result_record.update(self._dispatch_retrain(job))
            result_record["status"] = "completed"
        except Exception as exc:
            LOGGER.warning(
                "Heartbeat: re-optimization failed for %s/%s: %s",
                project_id,
                model_name,
                exc,
            )
            result_record["status"] = "failed"
            result_record["error"] = str(exc)

        return result_record

    def _dispatch_retrain(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to trigger re-optimization via the optimization cache invalidation path.

        For a full model re-run the caller must supply actual tensors/models.
        This lightweight path records the invalidation intent so the next
        ``agnitra.optimize()`` call picks up a fresh run.
        """
        from agnitra.core.runtime.cache import OptimizationCache

        sig = job.get("fingerprint_signature")
        if sig:
            cache = OptimizationCache()
            cache.invalidate(sig)
            LOGGER.info("Heartbeat: invalidated cache for signature %s", sig[:8] if sig else "?")
            return {"cache_invalidated": True, "signature": sig}
        return {"cache_invalidated": False}

    def _append_results(self, results: List[Dict[str, Any]]) -> None:
        try:
            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            with self.results_path.open("a", encoding="utf-8") as fh:
                for record in results:
                    fh.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            LOGGER.warning("Heartbeat: could not write results: %s", exc)


__all__ = ["OptimizationHeartbeat"]
