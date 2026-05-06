# Agnitra

**The inference optimizer for decoder-only LLMs. One line, no retraining, faster than `torch.compile` on the architectures you actually run in production.**

[![PyPI version](https://img.shields.io/pypi/v/agnitra?color=blue&label=PyPI)](https://pypi.org/project/agnitra/)
[![Python](https://img.shields.io/pypi/pyversions/agnitra)](https://pypi.org/project/agnitra/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

```python
import agnitra

result = agnitra.optimize(model, input_shape=(1, 512))
fast_model = result.optimized_model
```

That's it. No retraining. No graph rewrites by hand. No serving stack to adopt.

## Supported architectures

Agnitra is intentionally narrow. The wedge is **decoder-only LLMs** — Llama-class models that account for ~80% of LLM inference spend in production. Every fine-tune of every supported architecture inherits the optimization decisions of its base model via architecture fingerprinting, so "13 architectures supported" effectively means "the ~100K decoder-LM fine-tunes on HuggingFace."

| Architecture | `model_type` | Reference model | Status |
|---|---|---|---|
| Llama 1/2/3 | `llama` | `meta-llama/Meta-Llama-3-8B-Instruct` | ✅ tuned specialist |
| Mistral | `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | ✅ tuned specialist |
| Mixtral | `mixtral` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | ✅ tuned specialist |
| Qwen 2 / 2.5 | `qwen2` | `Qwen/Qwen2.5-7B-Instruct` | ✅ tuned specialist |
| Qwen 2 MoE | `qwen2_moe` | `Qwen/Qwen2.5-MoE` | ✅ tuned specialist |
| Gemma 1 / 2 | `gemma` / `gemma2` | `google/gemma-2-9b-it` | ✅ tuned specialist |
| Phi / Phi-3 | `phi` / `phi3` | `microsoft/Phi-3-mini-4k-instruct` | 🟡 generic decoder-LM |
| DeepSeek V2 | `deepseek_v2` | `deepseek-ai/DeepSeek-V2-Lite` | 🟡 generic decoder-LM |
| OLMo, Yi, Falcon | `olmo` / `yi` / `falcon` | `allenai/OLMo-7B` | 🟡 generic decoder-LM |
| Encoder transformers (BERT, RoBERTa, ViT) | — | — | ❌ pass-through |
| Image generation (SDXL, FLUX) | — | — | ❌ pass-through (ring 2) |
| Speech (Whisper) | — | — | ❌ pass-through (ring 3) |

When a model is outside the ring-1 set, `agnitra.optimize` returns the input model unchanged with `result.notes["passthrough"] = True` and the detected architecture string. **Honest scoping is a feature** — a silent 5% no-op speedup destroys customer trust faster than honest refusal.

LoRA fine-tunes are supported via `peft.merge_and_unload()` first; hot-swappable adapters are not yet supported.

## Roadmap rings

- **Ring 1 (now):** decoder-only LLMs. Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, etc.
- **Ring 2 (planned):** image generation. SDXL, SD3, FLUX. Different optimization landscape (UNet attention, classifier-free guidance batching, VAE decode).
- **Ring 3 (planned):** speech. Whisper, Wav2Vec2.
- **Out of scope:** encoder transformers, multimodal pipelines, image classification, training-time optimization, multi-GPU sharding.

## Status

**Beta.** The optimizer works end-to-end on real models. Public benchmark numbers vs. `torch.compile` / vLLM / TensorRT-LLM are pending the first H100 run — see [`benchmarks/llama3_h100/RESULTS.md`](benchmarks/llama3_h100/RESULTS.md). Until those numbers are published, treat any "2x faster" claim as unverified.

## Install

```bash
pip install agnitra
```

Optional extras:

```bash
pip install "agnitra[openai]"   # LLM-guided kernel suggestions via OpenAI
pip install "agnitra[rl]"       # PPO-guided search (Stable-Baselines3)
pip install "agnitra[nvml]"     # GPU telemetry via pynvml
```

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import agnitra

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.float16
).cuda()

result = agnitra.optimize(model, input_shape=(1, 512), enable_rl=False)
fast = result.optimized_model

# Use `fast` everywhere you used `model` before.
```

A complete runnable script lives at [`examples/quickstart.py`](examples/quickstart.py).

### Want a real speedup? Quantize.

The TF32 + SDPA + `torch.compile` defaults match what HuggingFace does today; the optimization that actually beats `transformers` baseline on Llama-class models is INT8 weight-only quantization:

```python
result = agnitra.optimize(
    model,
    input_shape=(1, 512),
    enable_rl=False,
    quantize="int8_weight",   # ← the lever
)
```

Expected impact on Llama-3-8B: ~1.3–1.7× throughput on memory-bound decode, ~2× memory reduction, near-zero quality loss. Requires `torchao` (`pip install "agnitra[quantize]"`).

## Integrations

### HuggingFace `transformers`

Replace `AutoModelForCausalLM` with `AgnitraModel`. Everything else stays identical:

```python
from agnitra.integrations.huggingface import AgnitraModel

model = AgnitraModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    agnitra_kwargs={"input_shape": (1, 512)},
).cuda()
# Use `model` like a normal transformers model — tokenizer, .generate(), logits.
```

Pass any `transformers.AutoModelFor...` class via `model_class=` for non-CausalLM workloads. See [`examples/quickstart_hf.py`](examples/quickstart_hf.py).

For an existing `transformers.pipeline()`:

```python
from agnitra.integrations.huggingface import optimize_pipeline

pipe = transformers.pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct")
optimize_pipeline(pipe, agnitra_kwargs={"input_shape": (1, 512)})
```

### `accelerate`

For users who go through `accelerate.Accelerator`, run Agnitra after `prepare()`:

```python
from accelerate import Accelerator
from agnitra.integrations.accelerate_helpers import optimize_after_prepare

accelerator = Accelerator()
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
model = optimize_after_prepare(model, input_shape=(1, 512))
```

## What Agnitra does

1. **Profiles** the model on real input shapes via `torch.profiler` + NVML telemetry.
2. **Suggests** kernel tuning parameters using either an LLM (OpenAI / Ollama) or a deterministic policy when no LLM is available.
3. **Applies** safe-by-default optimizations: TF32 matmul, FlashAttention / SDPA, `torch.compile` with the right mode, optional fused Triton kernels for matmul / layer-norm.
4. **Verifies** the optimized model produces the same outputs as the baseline.
5. **Returns** the patched `nn.Module` plus a structured report (`RuntimeOptimizationResult`).

## What Agnitra is *not*

Honest scope, so you don't waste a day:

- **Not a serving runtime.** It does not implement paged KV cache, continuous batching, or speculative decoding. Pair Agnitra with vLLM / TGI / your own serving stack.
- **Limited quantization (W8A16 only).** Agnitra supports INT8 weight-only quantization via `quantize="int8_weight"`, delegating to `torchao`. This is the optimization that beats plain HuggingFace + `torch.compile` (HF doesn't quantize by default). INT4 / activation quantization / AWQ / GPTQ are out of scope; if you have a model already quantized via those, Agnitra will optimize it but won't re-quantize.
- **Not a trainer.** Inference only. Training-time optimization is out of scope.
- **Not a multi-GPU sharder.** Single-GPU optimization. Tensor parallelism is a separate problem.

## Benchmarks

Reproducible benchmarks live in [`benchmarks/`](benchmarks/). The headline suite is [`benchmarks/llama3_h100/`](benchmarks/llama3_h100/) — a one-command repro comparing Agnitra to HuggingFace `transformers`, `torch.compile`, vLLM, and TensorRT-LLM on Llama-3-8B at batch sizes 1, 8, 32.

Five access paths are documented (Docker, host venv, Modal, Lambda Labs / RunPod SSH, GitHub Actions self-hosted). The cheapest is Modal:

```bash
pip install modal && modal token new
HF_TOKEN=hf_xxx modal run benchmarks/llama3_h100/modal_runner.py
```

`RESULTS.md` is regenerated by the benchmark CI workflow on every release tag and gates merges on a >5% throughput regression vs. the previous baseline.

### LangChain

Agents call the LLM hundreds of times per task. A 1.5x speedup on the model becomes a 1.5x reduction in the agent's wall-clock time.

```python
from langchain_huggingface import HuggingFacePipeline
from agnitra.integrations.langchain import optimize_llm

llm = HuggingFacePipeline.from_model_id("meta-llama/Meta-Llama-3-8B-Instruct", task="text-generation")
optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 512), "quantize": "int8_weight"})
# Use `llm` exactly as before — all chains/agents downstream get the speedup.
```

See [`examples/quickstart_langchain.py`](examples/quickstart_langchain.py).

### LlamaIndex

Same pattern for RAG and agent flows.

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from agnitra.integrations.llama_index import optimize_llm

llm = HuggingFaceLLM(model_name="meta-llama/Meta-Llama-3-8B-Instruct", ...)
optimize_llm(llm, agnitra_kwargs={"input_shape": (1, 512), "quantize": "int8_weight"})
```

See [`examples/quickstart_llama_index.py`](examples/quickstart_llama_index.py).

## CLI

```bash
agnitra optimize --model my_model.pt --input-shape 1,3,224,224 --output optimized.pt
agnitra optimize-dir --models-dir /var/agnitra/fleet  # batch-optimize a fine-tune farm
agnitra doctor                    # health check: torch / CUDA / NVML / Ollama / license
agnitra heartbeat --interval 30   # background re-optimization daemon
```

### Fine-tune farms (`agnitra optimize-dir`)

If you run 50+ fine-tuned variants of the same base model in production (one per customer, one per use-case), Agnitra's architecture-fingerprint cache means **the second model with the same architecture inherits the first's optimization decisions**. Optimizing 50 Llama-3-8B fine-tunes takes nearly the same wall time as optimizing one.

```bash
agnitra optimize-dir \
  --models-dir /var/agnitra/fleet \
  --quantize int8_weight \
  --input-shape 1,512
```

The directory layout is the standard HuggingFace shape — one subdirectory per model, each containing `config.json` plus weights.

## API server (optional)

If you want to call the optimizer remotely (CI workers, hosted inference):

```bash
agnitra-api    # binds to 127.0.0.1:8080 by default; AGNITRA_API_HOST overrides
```

The server exposes `POST /optimize`, `GET /jobs/{id}`, `GET /health`, and `WebSocket /ws/jobs/{id}` for live status. By default it listens on localhost; set `AGNITRA_ALLOW_PUBLIC_BIND=1` if you intentionally bind to a public interface.

## NVIDIA ecosystem

Agnitra drives traffic into NVIDIA's stack rather than competing with it. Most HuggingFace developers cannot use TensorRT-LLM directly because it requires C++ and a multi-step engine build. Agnitra wraps that:

```python
result = agnitra.optimize(
    model,
    backend="tensorrt_llm",
    backend_kwargs={"engine_dir": "./engine"},
)
```

For deployable containers, package an Agnitra-optimized model as a NIM-compatible Triton bundle:

```bash
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --target h100
```

Output is a Triton model repository plus a `Dockerfile` based on `nvcr.io/nvidia/tritonserver`. See [`docs/guides/nvidia`](docs/guides/nvidia.mdx) for engine build steps, NGC catalog publishing, and the [NVIDIA Inception](https://www.nvidia.com/startups/) program path.

## Configuration

| Environment variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | Enables the LLM-guided suggestion path. |
| `AGNITRA_OLLAMA_URL` | Local LLM backend (default `http://localhost:11434`). |
| `AGNITRA_API_HOST` / `AGNITRA_API_PORT` | API server bind interface. |
| `AGNITRA_LICENSE_PATH` | Path to a license file when using enterprise features. |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | POST optimization results to Slack / Discord / Telegram. |

## Repository layout

```
agnitra/
  sdk.py                    public optimize() entry point
  cli.py                    Click CLI
  _sdk/                     low-level optimizer (FX, kernels, RL)
  core/
    optimizer/              LLM- and RL-guided optimization
    runtime/                runtime patching, telemetry, fingerprinting, cache
    kernel/                 Triton kernel generation
    metering/, billing/     usage events and Stripe integration
    licensing/              license validation
    notifications/          webhook notifiers
  api/                      Starlette REST API server
benchmarks/                 reproducible benchmark suites
examples/                   small focused examples
tests/                      pytest suite
```

## Contributing

Bug reports, benchmark PRs, and "you're handicapping vLLM" issues are all welcome — the benchmark suite is meant to be adversarially reviewed. Open an issue or PR.

## License

[Apache 2.0](LICENSE).
