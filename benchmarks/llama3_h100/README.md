# Llama-3-8B on H100 — Reproducible Benchmark

Public, reproducible comparison of Agnitra against the strongest open-source
inference runtimes on a single workload.

## What this measures

| Dimension | Value |
|-----------|-------|
| Model | `meta-llama/Meta-Llama-3-8B-Instruct` (FP16) |
| Hardware | NVIDIA H100 80GB SXM5 |
| Input length | 512 tokens |
| Output length | 128 tokens (greedy decode) |
| Batch sizes | 1, 8, 32 |
| Runners | HuggingFace `transformers`, `torch.compile`, vLLM, TensorRT-LLM, **Agnitra** |
| Metrics | TTFT (p50, p99), decode tokens/sec, throughput tokens/sec, peak GPU memory |

## Why it exists

Vendor benchmarks are not credible. This directory exists so that a skeptical
engineer with an H100 can run **one command**, get the same numbers we publish
(within ±5%), and verify or refute Agnitra's performance claims.

If you find a configuration that handicaps a baseline, **open a PR**. We treat
fair-fight benchmarks as adversarial review, not marketing.

## Reproduce

### Prerequisites for every option

- An NVIDIA H100 (other GPUs run but the directory's name is a lie —
  see "What about other GPUs?" below).
- NVIDIA driver compatible with **CUDA 12.1** (driver ≥ 525.85.12).
- Roughly **80 GB free disk** (model weights + vLLM cache + optional
  TRT-LLM engine + Python deps).
- A HuggingFace token with access to
  [`meta-llama/Meta-Llama-3-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  (the model is gated; request access first).

### Option A — Docker (recommended)

The Dockerfile expects the build context at the **repo root** because it
installs Agnitra from source. Run the build from the repo root:

```bash
# from the repo root, NOT this directory
docker build -t agnitra-bench:llama3-h100 \
  -f benchmarks/llama3_h100/Dockerfile .

# then run, mounting raw/ so JSON outputs survive container removal
docker run --rm --gpus all \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$PWD/benchmarks/llama3_h100/raw:/work/benchmarks/llama3_h100/raw" \
  agnitra-bench:llama3-h100
```

### Option B — host machine

```bash
# from this directory
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ../..      # install Agnitra from the repo root
HF_TOKEN=hf_xxx ./run.sh
```

### Option C — Modal (serverless H100, no infra to manage)

If you don't want to provision a box, the included Modal wrapper rents
an H100 by the second:

```bash
pip install modal && modal token new
HF_TOKEN=hf_xxx modal run benchmarks/llama3_h100/modal_runner.py
```

The Modal job streams logs to your terminal and downloads `raw/` and
`RESULTS.md` back to your machine when done. See the file's docstring
for cost details.

### Option D — Lambda Labs / RunPod (raw SSH H100)

Many cloud providers ship Ubuntu images with a working CUDA driver but
no Python venv. Run the bootstrap helper, then `run.sh`:

```bash
ssh ubuntu@<your-h100-host>
git clone https://github.com/Agnitraai/Agnitraai.git
cd Agnitraai/benchmarks/llama3_h100
./bootstrap.sh             # installs Python 3.11 + deps + Agnitra
HF_TOKEN=hf_xxx ./run.sh
```

### Option E — GitHub Actions self-hosted runner

The `.github/workflows/benchmark.yml` workflow already exists. The
runner needs:

- A label set including `[self-hosted, gpu, h100]`
- Docker with the NVIDIA container runtime configured
- The repository secret `HF_TOKEN`

Trigger with `gh workflow run benchmark.yml` (or push a `v*` tag).

### What about other GPUs?

The directory's name is `llama3_h100` because that's the published
target. The runner code doesn't check the GPU SKU, so it will execute
on A100 / L40S / 4090 if one is present — but the resulting numbers
should not be compared to the published H100 baseline. `run.sh` prints
a loud warning when the detected GPU isn't an H100.

### What `run.sh` does

1. Logs `nvidia-smi` and library versions to `raw/env.txt`.
2. Runs each runner across batch sizes [1, 8, 32]. Each runner writes
   `raw/<runner>.json`. A failing runner (e.g. TensorRT-LLM not installed)
   does **not** abort the others.
3. Aggregates results into `RESULTS.md`.

Total wall time on a single H100: roughly 25–40 minutes (most of it
TensorRT-LLM engine build).

## File layout

```
benchmarks/llama3_h100/
  README.md           this file
  Dockerfile          pinned environment
  requirements.txt    pinned Python deps
  run.sh              one-command repro
  workload.py         frozen prompts and decode params
  schema.py           shared result dataclasses
  common.py           timing, memory, GPU helpers
  compare.py          aggregator → RESULTS.md
  RESULTS.md          published results table
  runners/
    hf.py
    torch_compile.py
    vllm_runner.py
    tensorrt_llm.py
    agnitra_runner.py
  raw/                JSON outputs (committed for verification)
```

## Methodology notes

- **TTFT** is measured by running `generate(max_new_tokens=1)` (or the
  runtime's equivalent of "decode one token") in isolation. This captures
  prefill cost cleanly.
- **Decode TPS** is `(output_len - 1) / (e2e_time - ttft_time)`.
- **Throughput TPS** is `(batch * output_len) / e2e_time` — what production
  serving cares about.
- **Warmup**: 3 throwaway iterations per (runner, batch_size) before
  measurement. `torch.compile` and TensorRT-LLM need this; including it for
  every runner keeps comparisons fair.
- **Measurement**: 10 iterations per (runner, batch_size). p50 and p99 are
  the 50th and 99th percentiles of those 10. (For p99 stability we
  recommend ≥30 iterations on a real publishing run; 10 is the CI default.)
- **Greedy decode** is used everywhere to remove sampling-induced variance.
- **Memory**: `torch.cuda.max_memory_allocated()` reset per measurement.

## Honest disclaimers

- vLLM and TensorRT-LLM are designed for *serving* workloads. Single-shot
  `generate()` is not their best mode. The numbers here are useful but a
  full picture also requires sustained-load benchmarks (coming).
- Agnitra is a graph-level optimizer; it does **not** implement paged KV
  cache or continuous batching. Expect Agnitra to be most competitive at
  low batch / low concurrency, and vLLM/TRT-LLM to widen at high batch.
- All numbers are *single-H100*. Multi-GPU tensor parallelism is a
  separate benchmark.

## Citing these numbers

Reference the commit hash, not the directory. Numbers in `RESULTS.md` are
regenerated by the benchmark CI workflow on every release tag.
