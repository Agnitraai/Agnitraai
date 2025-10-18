"""Background worker utilities for the hosted API."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)


OptimizationCallable = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]


@dataclass
class OptimizationJob:
    identifier: str
    payload: Dict[str, Any]
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class OptimizationQueue:
    """Asyncio-based background queue for optimization workloads."""

    def __init__(self, worker: OptimizationCallable, *, concurrency: int = 1) -> None:
        self.worker = worker
        self.concurrency = max(1, concurrency)
        self._jobs: Dict[str, OptimizationJob] = {}
        self._queue: Optional[asyncio.Queue[str]] = None
        self._tasks: list[asyncio.Task[None]] = []
        self._started = False

    def start(self) -> None:
        """Start worker tasks lazily."""

        if self._started:
            return
        loop = asyncio.get_event_loop()
        if self._queue is None:
            self._queue = asyncio.Queue()
        for _ in range(self.concurrency):
            task = loop.create_task(self._worker_loop())
            self._tasks.append(task)
        self._started = True

    async def enqueue(self, payload: Dict[str, Any]) -> OptimizationJob:
        job_id = uuid.uuid4().hex
        payload_with_id = dict(payload)
        payload_with_id.setdefault("_job_id", job_id)
        job = OptimizationJob(identifier=job_id, payload=payload_with_id)
        self._jobs[job_id] = job
        self.start()
        assert self._queue is not None  # for type checkers
        await self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Optional[OptimizationJob]:
        return self._jobs.get(job_id)

    async def _worker_loop(self) -> None:
        while True:
            if self._queue is None:
                await asyncio.sleep(0.01)
                continue
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                self._queue.task_done()
                continue
            job.status = "running"
            try:
                job.result = await self.worker(job.payload)
                job.status = "completed"
            except Exception as exc:  # pragma: no cover - defensive
                job.error = str(exc)
                job.status = "failed"
                logger.exception("Background optimization failed for %s", job_id)
            finally:
                self._queue.task_done()
