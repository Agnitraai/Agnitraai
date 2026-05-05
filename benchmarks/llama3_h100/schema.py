"""Shared result dataclasses.

All runners write JSON conforming to this schema. compare.py reads it.
The schema is intentionally flat and JSON-serializable — no numpy, no
torch tensors.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Latency:
    p50_ms: float
    p99_ms: float
    mean_ms: float
    samples_ms: List[float]


@dataclass
class BatchResult:
    batch_size: int
    input_tokens: int
    output_tokens: int
    ttft: Latency
    e2e: Latency
    decode_tps: float          # (output_tokens - 1) * batch_size / decode_time
    throughput_tps: float      # batch_size * output_tokens / e2e_time
    peak_memory_gb: float
    notes: str = ""


@dataclass
class RunnerResult:
    runner: str
    model_id: str
    device: str
    gpu_name: str
    torch_version: str
    cuda_version: str
    library_version: str       # e.g. transformers 4.44.2, vllm 0.5.4, agnitra 0.1.0
    timestamp_utc: str
    success: bool
    error: Optional[str] = None
    batches: List[BatchResult] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())


def load_runner_result(path: Path) -> RunnerResult:
    raw = json.loads(path.read_text())
    batches = [
        BatchResult(
            batch_size=b["batch_size"],
            input_tokens=b["input_tokens"],
            output_tokens=b["output_tokens"],
            ttft=Latency(**b["ttft"]),
            e2e=Latency(**b["e2e"]),
            decode_tps=b["decode_tps"],
            throughput_tps=b["throughput_tps"],
            peak_memory_gb=b["peak_memory_gb"],
            notes=b.get("notes", ""),
        )
        for b in raw.get("batches", [])
    ]
    return RunnerResult(
        runner=raw["runner"],
        model_id=raw["model_id"],
        device=raw["device"],
        gpu_name=raw["gpu_name"],
        torch_version=raw["torch_version"],
        cuda_version=raw["cuda_version"],
        library_version=raw["library_version"],
        timestamp_utc=raw["timestamp_utc"],
        success=raw["success"],
        error=raw.get("error"),
        batches=batches,
        extra=raw.get("extra", {}),
    )
