"""GPU fingerprinting and usage tracking for per-GPU licensing."""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class GpuFingerprint:
    """Structured GPU descriptor captured during a run."""

    uuid: str
    name: str
    memory_mb: Optional[int]
    driver_version: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "uuid": self.uuid,
            "name": self.name,
            "memory_mb": self.memory_mb,
            "driver_version": self.driver_version,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


def collect_gpu_fingerprints() -> Sequence[GpuFingerprint]:
    """Return fingerprints for GPUs visible to the runtime."""

    collectors = (_collect_with_nvml, _collect_with_nvidia_smi, _collect_cpu_fallback)
    for collector in collectors:
        fingerprints = collector()
        if fingerprints:
            return fingerprints
    return []


def _collect_with_nvml() -> Sequence[GpuFingerprint]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return []

    try:
        pynvml.nvmlInit()
    except Exception as exc:
        logger.debug("pynvml initialisation failed: %s", exc)
        return []

    try:
        device_count = pynvml.nvmlDeviceGetCount()
    except Exception as exc:
        logger.debug("pynvml device enumeration failed: %s", exc)
        pynvml.nvmlShutdown()
        return []

    fingerprints: List[GpuFingerprint] = []
    for index in range(device_count):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid_bytes = pynvml.nvmlDeviceGetUUID(handle)
            uuid = uuid_bytes.decode("utf-8") if isinstance(uuid_bytes, bytes) else str(uuid_bytes)
            name_bytes = pynvml.nvmlDeviceGetName(handle)
            name = name_bytes.decode("utf-8") if isinstance(name_bytes, bytes) else str(name_bytes)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_mb = int(mem_info.total / (1024 * 1024))
            driver = pynvml.nvmlSystemGetDriverVersion()
            driver_version = driver.decode("utf-8") if isinstance(driver, bytes) else str(driver)
            fingerprints.append(
                GpuFingerprint(
                    uuid=uuid,
                    name=name,
                    memory_mb=memory_mb,
                    driver_version=driver_version,
                    metadata={"source": "nvml", "index": index},
                )
            )
        except Exception as exc:
            logger.debug("Failed to collect NVML fingerprint for index %s: %s", index, exc)

    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
    return fingerprints


def _collect_with_nvidia_smi() -> Sequence[GpuFingerprint]:
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
        proc = subprocess.run(command, check=False, capture_output=True, text=True, timeout=5)
    except Exception as exc:
        logger.debug("nvidia-smi execution failed: %s", exc)
        return []

    if proc.returncode != 0:
        logger.debug("nvidia-smi returned non-zero exit code %s: %s", proc.returncode, proc.stderr)
        return []

    fingerprints: List[GpuFingerprint] = []
    for line in proc.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        uuid = parts[0]
        name = parts[1]
        memory_mb = None
        driver_version = None
        if len(parts) >= 3:
            memory_match = re.search(r"(\d+)", parts[2])
            if memory_match:
                memory_mb = int(memory_match.group(1))
        if len(parts) >= 4:
            driver_version = parts[3]
        fingerprints.append(
            GpuFingerprint(
                uuid=uuid,
                name=name,
                memory_mb=memory_mb,
                driver_version=driver_version,
                metadata={"source": "nvidia-smi"},
            )
        )
    return fingerprints


def _collect_cpu_fallback() -> Sequence[GpuFingerprint]:
    """Return empty list so CPU-only hosts do not count against GPU licenses."""

    return []


class GpuUsageTracker:
    """Tracks GPU usage for per-GPU licensing validation."""

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._usage: List[Dict[str, Any]] = []

    def register_run(
        self,
        *,
        license_key: str,
        org_id: str,
        fingerprints: Iterable[GpuFingerprint],
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc)
        seen = []
        for fingerprint in fingerprints:
            self._registry[fingerprint.uuid] = {
                "fingerprint": fingerprint.to_dict(),
                "last_seen": timestamp.isoformat(),
                "org_id": org_id,
                "license_key": license_key,
            }
            seen.append(fingerprint.to_dict())
        record = {
            "timestamp": timestamp.isoformat(),
            "license_key": license_key,
            "org_id": org_id,
            "active_gpu_count": len(self._registry),
            "fingerprints": seen,
        }
        self._usage.append(record)
        return record

    def usage_records(self) -> List[Dict[str, Any]]:
        return list(self._usage)

    def active_gpu_count(self) -> int:
        return len(self._registry)
