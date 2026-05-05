"""vLLM runner — the strongest open-source serving baseline.

vLLM has paged KV cache and continuous batching; for serving workloads
it is the bar to beat. We run it in single-shot offline mode to keep the
comparison apples-to-apples with the HF-style runners. A full sustained
serving comparison requires a different harness (coming separately).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

# Allow `python runners/vllm_runner.py` to import sibling modules.
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (  # noqa: E402
    cuda_memory_tracker,
    cuda_sync,
    gpu_info,
    latency_from_samples,
    utc_timestamp,
)
from schema import BatchResult, RunnerResult  # noqa: E402
from workload import WORKLOAD  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    gpu_name, cuda_version, torch_version = gpu_info()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", hf_token)

    try:
        import vllm
        from vllm import LLM, SamplingParams
        version = f"vllm {vllm.__version__}"
    except Exception as exc:
        RunnerResult(
            runner="vllm",
            model_id=WORKLOAD.model_id,
            device="cuda",
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            library_version="vllm (not installed)",
            timestamp_utc=utc_timestamp(),
            success=False,
            error=f"vllm import failed: {exc!r}",
        ).write(args.output)
        return 1

    try:
        # Single-shot, no chunked prefill, no speculative decoding —
        # we want a clean comparison, not a tuned bake-off.
        llm = LLM(
            model=WORKLOAD.model_id,
            dtype="float16",
            enforce_eager=False,
            # Cap KV cache so the largest batch doesn't OOM differently
            # than transformers — both share the H100 80GB envelope.
            gpu_memory_utilization=0.90,
            max_model_len=WORKLOAD.input_tokens + WORKLOAD.output_tokens + 64,
        )
    except Exception as exc:
        RunnerResult(
            runner="vllm",
            model_id=WORKLOAD.model_id,
            device="cuda",
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            library_version=version,
            timestamp_utc=utc_timestamp(),
            success=False,
            error=f"vllm engine init failed: {exc!r}\n{traceback.format_exc()}",
        ).write(args.output)
        return 2

    sampling_full = SamplingParams(
        temperature=WORKLOAD.temperature,
        top_p=WORKLOAD.top_p,
        top_k=WORKLOAD.top_k,
        max_tokens=WORKLOAD.output_tokens,
        min_tokens=WORKLOAD.output_tokens,
    )
    sampling_first = SamplingParams(
        temperature=WORKLOAD.temperature,
        top_p=WORKLOAD.top_p,
        top_k=WORKLOAD.top_k,
        max_tokens=1,
        min_tokens=1,
    )

    batches: list[BatchResult] = []
    overall_error: str | None = None

    try:
        for batch_size in WORKLOAD.batch_sizes:
            prompts = WORKLOAD.prompts(batch_size)

            for _ in range(WORKLOAD.warmup_iters):
                llm.generate(prompts, sampling_full, use_tqdm=False)

            ttft_samples: list[float] = []
            e2e_samples: list[float] = []
            with cuda_memory_tracker() as mem_holder:
                for _ in range(WORKLOAD.measure_iters):
                    cuda_sync()
                    t0 = time.perf_counter()
                    llm.generate(prompts, sampling_first, use_tqdm=False)
                    cuda_sync()
                    ttft_samples.append((time.perf_counter() - t0) * 1000.0)

                for _ in range(WORKLOAD.measure_iters):
                    cuda_sync()
                    t0 = time.perf_counter()
                    llm.generate(prompts, sampling_full, use_tqdm=False)
                    cuda_sync()
                    e2e_samples.append((time.perf_counter() - t0) * 1000.0)

            ttft = latency_from_samples(ttft_samples)
            e2e = latency_from_samples(e2e_samples)
            decode_time_ms = max(e2e.mean_ms - ttft.mean_ms, 1e-6)
            decode_tps = (
                (WORKLOAD.output_tokens - 1) * batch_size
                / (decode_time_ms / 1000.0)
            )
            throughput_tps = (
                batch_size * WORKLOAD.output_tokens / (e2e.mean_ms / 1000.0)
            )

            batches.append(BatchResult(
                batch_size=batch_size,
                input_tokens=WORKLOAD.input_tokens,
                output_tokens=WORKLOAD.output_tokens,
                ttft=ttft,
                e2e=e2e,
                decode_tps=decode_tps,
                throughput_tps=throughput_tps,
                peak_memory_gb=mem_holder[0],
            ))
    except Exception as exc:
        overall_error = f"{exc!r}\n{traceback.format_exc()}"

    RunnerResult(
        runner="vllm",
        model_id=WORKLOAD.model_id,
        device="cuda",
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        library_version=version,
        timestamp_utc=utc_timestamp(),
        success=overall_error is None and bool(batches),
        error=overall_error,
        batches=batches,
        extra={"engine_mode": "offline_single_shot"},
    ).write(args.output)
    return 0 if overall_error is None else 3


if __name__ == "__main__":
    sys.exit(main())
