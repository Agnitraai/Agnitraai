# Llama-3-8B / H100 Benchmark — Results

Reproducible numbers from this directory's `run.sh`. Hand edits
are overwritten — change `compare.py` to change formatting.

## Workload

- Model: `meta-llama/Meta-Llama-3-8B-Instruct`
- Input tokens: 512
- Output tokens: 128 (greedy decode)
- Warmup / measure iterations: 3 / 10

## Results

### Batch size 1

| Runner | TTFT p50 | TTFT p99 | Throughput | Decode | Peak Mem |
|---|---:|---:|---:|---:|---:|
| HuggingFace `transformers` | 23.7 ms | 23.9 ms | 53.5 tok/s | 53.7 tok/s | 15.4 GB |
| `torch.compile` | 23.5 ms | 23.8 ms | 52.2 tok/s | 52.3 tok/s | 15.4 GB |
| vLLM | — | — | — | — | — |  *(vllm import failed: ModuleNotFoundError("No module named 'vllm'"))*
| TensorRT-LLM | — | — | — | — | — |  *(tensorrt_llm import failed: ModuleNotFoundError("No module named 'tensorrt_llm.r)*
| **Agnitra** | — | — | — | — | — |  *(model load/prepare failed: LicenseValidationError('License file not found at /ro)*

### Batch size 8

| Runner | TTFT p50 | TTFT p99 | Throughput | Decode | Peak Mem |
|---|---:|---:|---:|---:|---:|
| HuggingFace `transformers` | 146 ms | 153 ms | 394 tok/s | 415 tok/s | 18.5 GB |
| `torch.compile` | 146 ms | 152 ms | 384 tok/s | 403 tok/s | 18.5 GB |
| vLLM | — | — | — | — | — |  *(vllm import failed: ModuleNotFoundError("No module named 'vllm'"))*
| TensorRT-LLM | — | — | — | — | — |  *(tensorrt_llm import failed: ModuleNotFoundError("No module named 'tensorrt_llm.r)*
| **Agnitra** | — | — | — | — | — |  *(model load/prepare failed: LicenseValidationError('License file not found at /ro)*

### Batch size 32

| Runner | TTFT p50 | TTFT p99 | Throughput | Decode | Peak Mem |
|---|---:|---:|---:|---:|---:|
| HuggingFace `transformers` | 577 ms | 586 ms | 882 tok/s | 1000 tok/s | 28.9 GB |
| `torch.compile` | 577 ms | 584 ms | 887 tok/s | 1006 tok/s | 28.9 GB |
| vLLM | — | — | — | — | — |  *(vllm import failed: ModuleNotFoundError("No module named 'vllm'"))*
| TensorRT-LLM | — | — | — | — | — |  *(tensorrt_llm import failed: ModuleNotFoundError("No module named 'tensorrt_llm.r)*
| **Agnitra** | — | — | — | — | — |  *(model load/prepare failed: LicenseValidationError('License file not found at /ro)*

## Speedup

_Agnitra runner did not produce results — speedup table omitted._

## Environment

- GPU: `NVIDIA H100 80GB HBM3`
- CUDA: `12.1`
- PyTorch: `2.4.0`
- Generated: `2026-05-06T01:06:51+00:00`

## Raw data

All per-runner JSON results are committed at `benchmarks/llama3_h100/raw/`.
Re-run `python compare.py` after editing the JSON to regenerate this file.
