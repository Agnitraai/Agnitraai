"""Shared loop for runners that use HuggingFace ``transformers`` semantics.

The HF, torch.compile, and Agnitra runners share the same generation loop;
only the model preparation step differs. Centralizing the loop here
guarantees they measure identical things.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Callable

# Allow `python runners/hf.py` to import sibling modules.
_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import (  # noqa: E402  (sys.path modified above)
    cuda_memory_tracker,
    gpu_info,
    latency_from_samples,
    time_block_ms,
    utc_timestamp,
)
from schema import BatchResult, RunnerResult  # noqa: E402
from workload import WORKLOAD  # noqa: E402

ModelPrep = Callable[["AutoModelForCausalLM"], "AutoModelForCausalLM"]  # noqa: F821


def parse_args(default_runner: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the runner's JSON result.")
    parser.add_argument("--runner-name", default=default_runner)
    parser.add_argument("--hf-token", default=None,
                        help="HuggingFace token (else reads HF_TOKEN env).")
    return parser.parse_args()


def run_hf_style(
    runner_name: str,
    library_version: str,
    prepare_model: ModelPrep,
    output_path: Path,
    hf_token: str | None,
    extra: dict | None = None,
) -> int:
    """Execute the standard HF-style benchmark and write the result file.

    ``prepare_model`` receives the loaded ``AutoModelForCausalLM`` and
    returns whatever object should be used for generation (typically the
    same model, possibly compiled or wrapped). The wrapped object must
    expose ``.generate(input_ids, attention_mask, **gen_kwargs)``.
    """
    gpu_name, cuda_version, torch_version = gpu_info()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - import-time failure
        RunnerResult(
            runner=runner_name,
            model_id=WORKLOAD.model_id,
            device="cpu",
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            library_version=library_version,
            timestamp_utc=utc_timestamp(),
            success=False,
            error=f"import failed: {exc!r}",
            extra=extra or {},
        ).write(output_path)
        return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        RunnerResult(
            runner=runner_name,
            model_id=WORKLOAD.model_id,
            device="cpu",
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            library_version=library_version,
            timestamp_utc=utc_timestamp(),
            success=False,
            error="CUDA not available; this benchmark requires a GPU.",
            extra=extra or {},
        ).write(output_path)
        return 2

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            WORKLOAD.model_id, token=hf_token, padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            WORKLOAD.model_id,
            token=hf_token,
            torch_dtype=torch.float16,
        ).to(device)
        model.eval()
        runtime = prepare_model(model)
    except Exception as exc:
        RunnerResult(
            runner=runner_name,
            model_id=WORKLOAD.model_id,
            device=device,
            gpu_name=gpu_name,
            torch_version=torch_version,
            cuda_version=cuda_version,
            library_version=library_version,
            timestamp_utc=utc_timestamp(),
            success=False,
            error=f"model load/prepare failed: {exc!r}\n{traceback.format_exc()}",
            extra=extra or {},
        ).write(output_path)
        return 3

    gen_common = dict(
        do_sample=False,
        temperature=WORKLOAD.temperature,
        top_p=WORKLOAD.top_p,
        top_k=WORKLOAD.top_k,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    batches: list[BatchResult] = []
    overall_error: str | None = None

    with torch.inference_mode():
        for batch_size in WORKLOAD.batch_sizes:
            try:
                prompts = WORKLOAD.prompts(batch_size)
                tokens = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=WORKLOAD.input_tokens,
                ).to(device)

                def _generate(max_new_tokens: int) -> None:
                    runtime.generate(
                        input_ids=tokens.input_ids,
                        attention_mask=tokens.attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=max_new_tokens,
                        **gen_common,
                    )

                # Warmup — combined prefill + a few decode steps so kernels
                # for both phases are compiled.
                for _ in range(WORKLOAD.warmup_iters):
                    _generate(WORKLOAD.output_tokens)

                with cuda_memory_tracker() as mem_holder:
                    ttft_samples = time_block_ms(
                        lambda: _generate(1), WORKLOAD.measure_iters
                    )
                    e2e_samples = time_block_ms(
                        lambda: _generate(WORKLOAD.output_tokens),
                        WORKLOAD.measure_iters,
                    )

                ttft = latency_from_samples(ttft_samples)
                e2e = latency_from_samples(e2e_samples)
                # Decode-only time per call.
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
            except torch.cuda.OutOfMemoryError as exc:
                batches.append(BatchResult(
                    batch_size=batch_size,
                    input_tokens=WORKLOAD.input_tokens,
                    output_tokens=WORKLOAD.output_tokens,
                    ttft=latency_from_samples([]),
                    e2e=latency_from_samples([]),
                    decode_tps=0.0,
                    throughput_tps=0.0,
                    peak_memory_gb=0.0,
                    notes=f"OOM at batch={batch_size}: {exc}",
                ))
                # Free what we can, continue to next batch size.
                torch.cuda.empty_cache()
            except Exception as exc:
                overall_error = (
                    f"batch={batch_size} failed: {exc!r}\n{traceback.format_exc()}"
                )
                break

    RunnerResult(
        runner=runner_name,
        model_id=WORKLOAD.model_id,
        device=device,
        gpu_name=gpu_name,
        torch_version=torch_version,
        cuda_version=cuda_version,
        library_version=library_version,
        timestamp_utc=utc_timestamp(),
        success=overall_error is None and any(b.e2e.samples_ms for b in batches),
        error=overall_error,
        batches=batches,
        extra=extra or {},
    ).write(output_path)
    return 0
