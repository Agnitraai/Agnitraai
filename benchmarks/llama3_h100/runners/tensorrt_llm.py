"""TensorRT-LLM runner — the NVIDIA-blessed performance ceiling.

TensorRT-LLM requires a separately built engine (`trtllm-build`) and is
sensitive to driver/CUDA versions. This runner does NOT build the
engine; it expects one to be present at the path given by
``TRTLLM_ENGINE_DIR``. If TRT-LLM or the engine is missing the runner
writes a "not installed" JSON result and exits cleanly so it does not
abort the rest of `run.sh`.

To build the engine on the same H100:

    pip install tensorrt_llm
    git clone https://github.com/NVIDIA/TensorRT-LLM
    cd TensorRT-LLM/examples/llama
    python convert_checkpoint.py \\
        --model_dir <local_llama3_8b_dir> \\
        --output_dir ./ckpt --dtype float16
    trtllm-build --checkpoint_dir ./ckpt \\
        --output_dir ./engine \\
        --gemm_plugin float16 \\
        --max_input_len 512 --max_seq_len 640 \\
        --max_batch_size 32

Then:

    export TRTLLM_ENGINE_DIR=$PWD/engine
    python runners/tensorrt_llm.py --output raw/tensorrt_llm.json
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

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


def _skip(args, gpu_name, cuda_version, torch_version, version, reason: str) -> int:
    RunnerResult(
        runner="tensorrt_llm",
        model_id=WORKLOAD.model_id,
        device="cuda",
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        library_version=version,
        timestamp_utc=utc_timestamp(),
        success=False,
        error=reason,
    ).write(args.output)
    return 0  # zero so run.sh keeps going


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--engine-dir", default=os.environ.get("TRTLLM_ENGINE_DIR"))
    parser.add_argument("--hf-token", default=None)
    args = parser.parse_args()

    gpu_name, cuda_version, torch_version = gpu_info()

    try:
        import tensorrt_llm  # noqa: F401
        from tensorrt_llm.runtime import ModelRunner
        version = f"tensorrt_llm {tensorrt_llm.__version__}"
    except Exception as exc:
        return _skip(args, gpu_name, cuda_version, torch_version,
                     "tensorrt_llm (not installed)",
                     f"tensorrt_llm import failed: {exc!r}")

    if not args.engine_dir or not Path(args.engine_dir).is_dir():
        return _skip(args, gpu_name, cuda_version, torch_version, version,
                     "TRTLLM_ENGINE_DIR not set or directory missing — "
                     "see runner docstring for engine build instructions.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            WORKLOAD.model_id, token=hf_token, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:
        return _skip(args, gpu_name, cuda_version, torch_version, version,
                     f"tokenizer load failed: {exc!r}")

    try:
        runner = ModelRunner.from_dir(
            engine_dir=args.engine_dir,
            rank=0,
        )
    except Exception as exc:
        return _skip(args, gpu_name, cuda_version, torch_version, version,
                     f"engine load failed: {exc!r}\n{traceback.format_exc()}")

    batches: list[BatchResult] = []
    overall_error: str | None = None

    import torch
    try:
        for batch_size in WORKLOAD.batch_sizes:
            prompts = WORKLOAD.prompts(batch_size)
            tokens = tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=WORKLOAD.input_tokens,
            )
            input_ids = [row.cuda() for row in tokens.input_ids]

            def _generate(max_new_tokens: int) -> None:
                runner.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    end_id=tokenizer.eos_token_id,
                    pad_id=tokenizer.pad_token_id,
                    temperature=WORKLOAD.temperature,
                    top_k=WORKLOAD.top_k,
                    top_p=WORKLOAD.top_p,
                )

            for _ in range(WORKLOAD.warmup_iters):
                _generate(WORKLOAD.output_tokens)

            ttft_samples: list[float] = []
            e2e_samples: list[float] = []
            with cuda_memory_tracker() as mem_holder:
                for _ in range(WORKLOAD.measure_iters):
                    cuda_sync()
                    t0 = time.perf_counter()
                    _generate(1)
                    cuda_sync()
                    ttft_samples.append((time.perf_counter() - t0) * 1000.0)
                for _ in range(WORKLOAD.measure_iters):
                    cuda_sync()
                    t0 = time.perf_counter()
                    _generate(WORKLOAD.output_tokens)
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
        runner="tensorrt_llm",
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
        extra={"engine_dir": args.engine_dir},
    ).write(args.output)
    return 0 if overall_error is None else 3


if __name__ == "__main__":
    sys.exit(main())
