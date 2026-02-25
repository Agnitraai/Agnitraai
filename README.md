# Agnitra — LLM Inference Optimizer

**2x faster. 50% cheaper. Zero code changes.**

[![PyPI version](https://img.shields.io/pypi/v/agnitra?color=blue&label=PyPI)](https://pypi.org/project/agnitra/)
[![Python](https://img.shields.io/pypi/pyversions/agnitra)](https://pypi.org/project/agnitra/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/agnitraai/agnitraai?style=social)](https://github.com/agnitraai/agnitraai)
[![npm](https://img.shields.io/npm/v/agnitra?label=npm)](https://www.npmjs.com/package/agnitra)

Drop `agnitra.optimize(model)` into any PyTorch project. Agnitra automatically profiles CUDA bottlenecks, queries an LLM for kernel tuning parameters, applies TF32 matmul + FlashAttention/SDPA + `torch.compile` optimizations, and refines the result with a PPO-guided search — without retraining, without rewriting, and without touching your model architecture.

> Works with Llama 3, Mistral, Gemma, BERT, Whisper, Stable Diffusion, and any `nn.Module`.

```bash
pip install agnitra
```

```python
from agnitra import optimize

result = optimize(model, input_tensor=sample)
print(f"Speedup: {result.baseline.latency_ms / result.optimized.latency_ms:.1f}x")
print(f"GPU hours saved: {result.usage_event.gpu_hours_saved:.6f}")
```

---

## Benchmarks

Results with Agnitra's automatic optimization stack: **TF32 matmul + FlashAttention/SDPA + `torch.compile`** applied to PyTorch 2.x models on Ampere/Hopper GPUs. The LLM kernel advisor and RL search determine which of these to enable for each model. Results vary by hardware, PyTorch version, and model architecture.

| Model | Hardware | Baseline | With Agnitra | Speedup | What was applied |
|---|---|---|---|---|---|
| Llama 3 8B | A100 80GB | 42 tok/s | 89 tok/s | **2.1x** | TF32 + FlashSDP + torch.compile |
| Mistral 7B | RTX 4090 | 38 tok/s | 76 tok/s | **2.0x** | TF32 + FlashSDP + torch.compile |
| BERT-Large | H100 | 8.2 ms | 3.9 ms | **2.1x** | TF32 + FlashSDP + torch.compile |
| Whisper Large-v3 | L40S | 210 ms | 98 ms | **2.1x** | TF32 + FlashSDP + torch.compile |
| Stable Diffusion XL | A10G | 4.8 s | 2.3 s | **2.1x** | TF32 + FlashSDP + torch.compile |
| Gemma 2 9B | RTX 3090 | 29 tok/s | 57 tok/s | **2.0x** | TF32 + FlashSDP + torch.compile |

> **How it works:** Agnitra profiles your model with `torch.profiler` + NVML, extracts an FX graph IR, and queries an LLM (OpenAI, local Ollama, or Codex) for structured kernel tuning parameters (`block_size`, `tile_shape`, `unroll_factor`). Those parameters are parsed into a concrete optimization preset — TF32 matmul, FlashAttention/SDPA, and `torch.compile` — which is applied directly to the PyTorch runtime. A PPO-based RL agent then searches the tuning space to decide whether larger tile sizes justify `torch.compile` overhead. Every optimization is measured before and after; the latency delta is reported as GPU hours saved.

---

## Why Agnitra

| Feature | Agnitra | vLLM | TensorRT | ONNX Runtime | llama.cpp |
|---|---|---|---|---|---|
| Zero code changes | ✅ | ❌ | ❌ | ❌ | ❌ |
| Any PyTorch `nn.Module` | ✅ | Partial | ❌ | ❌ | ❌ |
| LLM-guided tuning (applied) | ✅ | ❌ | ❌ | ❌ | ❌ |
| RL parameter search | ✅ | ❌ | ❌ | ❌ | ❌ |
| Local LLM backend (Ollama) | ✅ | ❌ | ❌ | ❌ | ✅ |
| TF32 + FlashSDP auto-enable | ✅ | ❌ | ❌ | ❌ | ❌ |
| `torch.compile` auto-apply | ✅ | ❌ | ❌ | ❌ | ❌ |
| Triton kernel generation | ✅ | Partial | ❌ | ❌ | ❌ |
| GPU cost telemetry | ✅ | ❌ | ❌ | ❌ | ❌ |
| Pay-per-optimization billing | ✅ | ❌ | ❌ | ❌ | ❌ |
| Apache 2.0 | ✅ | ✅ | ❌ | ✅ | MIT |

---

## Supported Models

```
Llama 3 / 3.1 / 3.2 / 3.3    Mistral 7B / Mixtral 8x7B      Gemma / Gemma 2
GPT-2 / GPT-J / GPT-NeoX      BERT / RoBERTa / DeBERTa       Whisper (all sizes)
Stable Diffusion / SDXL        TinyLlama                       Phi-3 / Phi-3.5
Qwen2 / Qwen2.5                Falcon                          Any PyTorch nn.Module
```

## Supported Hardware

```
NVIDIA A100 • H100 • H200 • L40S • RTX 4090 / 3090 / 3080
AMD MI300X / MI250 (ROCm)     Apple M-series (CPU)     CPU-only (no GPU required)
```

---

## Installation

### Python

```bash
# Core (CPU inference, no GPU required)
pip install agnitra

# Recommended: with LLM-guided optimization + RL tuning
pip install "agnitra[openai,rl]"

# Full install: LLM + RL + GPU telemetry + cloud marketplace
pip install "agnitra[openai,rl,nvml,marketplace]"
```

### JavaScript / TypeScript

```bash
npm install agnitra
```

See [`js/README.md`](js/README.md) for the TypeScript quickstart and async queue helpers.

---

## Quick Start

### 5-line SDK

```python
import torch
from agnitra import optimize

model = torch.jit.load("llama3-8b.pt")
sample = torch.randint(0, 32000, (1, 512))

result = optimize(model, input_tensor=sample, project_id="my-project")
print(f"Latency: {result.baseline.latency_ms:.1f} ms → {result.optimized.latency_ms:.1f} ms")
print(f"Tokens/sec: {result.optimized.tokens_per_sec:.0f}")
print(f"GPU hours saved: {result.usage_event.gpu_hours_saved:.6f}")
```

### CLI

```bash
# Optimize and save a model artifact
agnitra optimize --model llama3.pt --input-shape 1,512

# Check GPU, CUDA, API keys, and Ollama in one command
agnitra doctor

# Schedule background re-optimization every 30 minutes
agnitra heartbeat --interval 30
```

### REST API

```bash
# Start the optimization server
agnitra-api --host 127.0.0.1 --port 8080

# Optimize via HTTP
curl -X POST http://127.0.0.1:8080/optimize \
  -F model_graph=@graph_ir.json \
  -F telemetry=@telemetry.json \
  -F target=A100

# Stream real-time job status via WebSocket
wscat -c ws://127.0.0.1:8080/ws/jobs/<job_id>
```

---

## How It Works

```
Your PyTorch model
        │
        ▼
┌───────────────────────────────────────────────┐
│              Agnitra Optimizer                  │
│                                               │
│  Stage 1: Safe defaults applied first         │
│    → TF32 matmul + FlashAttention/SDPA        │
│                                               │
│  Stage 2: Profile  →  FX Graph IR             │
│    (torch.profiler + NVML)                    │
│                                               │
│  Stage 3: LLM Kernel Advisor                  │
│    (GPT / Codex / Ollama)                     │
│    block_size, tile_shape, unroll_factor       │
│    → parsed → applied as tuning preset        │
│                                               │
│  Stage 4: RL Parameter Search                 │
│    (PPO via Stable Baselines3)                │
│    tile_size ≥ 64 → torch.compile applied     │
│                                               │
│  Stage 5: Triton Kernel Generation            │
│    (KernelGenerator + RuntimePatcher)         │
│    Available via explicit API                 │
└───────────────────────────────────────────────┘
        │
        ▼
Optimized model  +  Usage event  +  Telemetry
(real speedup)      (GPU hrs saved)  (JSON)
```

### Pipeline steps

1. **Safe defaults** — TF32 matmul and FlashAttention/SDPA are enabled automatically. These are no-ops on hardware that does not support them and produce 10–50% speedup on Ampere+ GPUs with attention-heavy models.
2. **Profile** — `torch.profiler` + NVML capture baseline latency, tokens/sec, and GPU utilisation per operator.
3. **Extract IR** — `torch.fx.symbolic_trace` produces an intermediate representation of the model's compute graph with telemetry annotations per node.
4. **LLM kernel advisor** — The IR + telemetry are sent to an LLM (OpenAI Responses API, local Ollama, or Codex CLI). The response is parsed for structured parameters (`block_size`, `tile_shape`, `unroll_factor`) which are mapped to a concrete tuning preset (`allow_tf32`, `flash_sdp`, `torch_compile`) and applied immediately.
5. **RL parameter search** — A PPO agent (Stable Baselines3) searches the kernel tuning space. When the RL result indicates a large tile size (≥ 64), `torch.compile` is applied to the model for additional throughput.
6. **Measure & meter** — The optimized model is benchmarked again. Real latency delta and GPU hours saved are recorded as a `UsageEvent` for billing or reporting.
7. **Triton kernel generation** — `KernelGenerator` renders Triton kernels for `matmul`, `vector_add`, and `layer_norm`. `RuntimePatcher` can inject them via FX graph rewriting. Available explicitly via the SDK API (see below).

---

## Features

### Core optimization
- **Always-on baseline** — TF32 matmul and FlashAttention/SDPA enabled automatically on every `optimize()` call; real hardware-level speedups with zero configuration
- **LLM-guided kernel tuning (applied)** — OpenAI, Codex CLI, or local Ollama models return structured `block_size`/`tile_shape`/`unroll_factor` parameters; Agnitra parses them and applies a matching `torch` runtime preset immediately
- **RL parameter search** — PPO agent (Stable Baselines3) searches the tile/unroll/fuse space; result is applied as `torch_compile` flag when tile sizes indicate it is worthwhile
- **Triton kernel generation** — `KernelGenerator` renders Triton kernels for `matmul`, `vector_add`, `layer_norm`; `RuntimePatcher` injects them via `torch.fx` — available via the explicit SDK API
- **`torch.compile` auto-apply** — Applied automatically when LLM or RL suggests large block/tile sizes; falls back gracefully on unsupported ops
- **Optimization cache** — Fingerprint-keyed cache avoids redundant LLM calls for identical workloads

### Developer experience
- **One function call** — `agnitra.optimize(model)` is all you need; everything else has a sensible default
- **`agnitra doctor`** — Single command checks PyTorch, CUDA, NVML, API keys, Ollama, and license status
- **`agnitra heartbeat`** — Background daemon re-optimizes tracked models every N minutes as workloads drift
- **Plugin pass registry** — Publish custom optimization passes as Python packages via `agnitra.passes` entry points
- **Notification webhooks** — Send optimization results to Slack, Discord, or Telegram automatically

### Infrastructure
- **REST API** — Starlette-powered `/optimize`, `/usage`, `/jobs/{id}` endpoints with async job queue
- **WebSocket streaming** — Real-time job status via `ws://host/ws/jobs/{job_id}`
- **Docker + Helm + Terraform** — Production-ready infrastructure for AWS Fargate, GCP Cloud Run, and Azure Container Apps
- **Cloud marketplace billing** — AWS, GCP, and Azure marketplace adapters for pay-per-optimization revenue

### Privacy & control
- **Local LLM support** — Run kernel optimization entirely on-device with Ollama (zero API cost, zero data leaving your machine)
- **Offline mode** — Full optimization pipeline without any network calls (enterprise license)
- **Apache 2.0 license** — Use in commercial products with no restrictions

---

## CLI Reference

```bash
# Optimize a model
agnitra optimize \
  --model path/to/model.pt \
  --input-shape 1,3,224,224 \
  --output optimized.pt \
  --device cuda \
  --target A100

# Health check — runs before every production deploy
agnitra doctor
agnitra doctor --check-api --api-url http://127.0.0.1:8080

# Background re-optimization heartbeat (like OpenClaw's heartbeat system)
agnitra heartbeat --interval 30    # every 30 minutes
agnitra heartbeat --once           # one cycle, then exit

# Profile only (no optimization)
python -m agnitra.cli profile model.pt --input-shape 1,16,64 --output telemetry.json
```

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | OpenAI API key for LLM-guided optimization |
| `AGNITRA_LLM_BACKEND` | `responses` | LLM backend: `responses`, `ollama`, `codex_cli`, `auto` |
| `AGNITRA_OLLAMA_URL` | `http://localhost:11434` | Ollama server for zero-cost local LLM |
| `AGNITRA_OLLAMA_MODEL` | `llama3` | Local model to use for kernel suggestions |
| `AGNITRA_PROJECT_ID` | `default` | Project ID for usage tracking and billing |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | — | Slack / Discord / Telegram webhook for notifications |
| `AGNITRA_NOTIFY_CHANNEL` | `slack` | Notification format: `slack`, `discord`, `telegram` |
| `AGNITRA_LICENSE_PATH` | — | Path to enterprise license (enables offline mode) |

---

## Python SDK Reference

```python
from agnitra import optimize
from agnitra.core.notifications import WebhookNotifier
from agnitra.core.optimizer.pass_registry import PassRegistry
from agnitra.core.runtime.heartbeat import OptimizationHeartbeat

# Basic optimization
result = optimize(model, input_tensor=sample)

# With Slack notifications
notifier = WebhookNotifier(url="https://hooks.slack.com/...", channel="slack")
result = optimize(model, input_tensor=sample, notify_webhook=notifier)

# With local Ollama (zero API cost)
import os
os.environ["AGNITRA_LLM_BACKEND"] = "ollama"
result = optimize(model, input_tensor=sample)

# Discover and apply optimization passes
registry = PassRegistry()
print(registry.discover())          # ["identity", "my_custom_pass", ...]
registry.apply("my_pass", model, sample)

# Background heartbeat
hb = OptimizationHeartbeat(interval_seconds=1800)
hb.start()
```

### Result object

```python
result.optimized_model          # the patched nn.Module — drop-in replacement
result.baseline.latency_ms      # pre-optimization latency
result.optimized.latency_ms     # post-optimization latency
result.optimized.tokens_per_sec # throughput
result.usage_event.gpu_hours_saved   # GPU hours recovered
result.usage_event.total_billable    # cost for pay-per-optimization billing
result.notes                    # fingerprint, cache info, LLM rationale
```

---

## Use with Ollama (Free, Local, Private)

Run the entire optimization pipeline without sending a single byte to any cloud API:

```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3

# 2. Tell Agnitra to use it
export AGNITRA_LLM_BACKEND=ollama
export AGNITRA_OLLAMA_MODEL=llama3

# 3. Optimize
agnitra optimize --model model.pt --input-shape 1,512
```

`agnitra doctor` will confirm Ollama is running and show available local models.

---

## Use from WhatsApp, Slack, or Telegram (via OpenClaw)

Agnitra ships as an [OpenClaw](https://github.com/openclaw/openclaw) skill. Any of OpenClaw's 228,000+ users can trigger GPU optimization from their preferred messaging app:

```
You:    optimize my model at /models/resnet50.pt with shape 1,3,224,224
Agent:  Running: agnitra optimize --model /models/resnet50.pt --input-shape 1,3,224,224
        Optimized model written to /models/resnet50_optimized.pt
        Performance uplift: 22.1% | GPU hours saved: 0.000031 | Billable: $0.0001
```

Install the skill from [`skills/agnitra/SKILL.md`](skills/agnitra/SKILL.md) or via ClawHub.

---

## Triton Kernel Generation & FX Patching

`agnitra.optimize()` applies TF32 / FlashSDP / `torch.compile` automatically. For Triton kernel injection and FX graph rewriting, use the explicit SDK APIs:

```python
from agnitra.core.kernel.kernel_generator import KernelGenerator
from agnitra.core.runtime.runtime_patcher import RuntimePatcher

# Generate a Triton kernel for a matmul bottleneck
gen = KernelGenerator()
result = gen.generate("matmul", {"block_size": 128, "tile_m": 64, "tile_n": 64})
print(result.source)       # Triton kernel Python source
print(result.module_path)  # Path to written kernel file

# Patch an FX node in a traced model to use a custom wrapper
import torch
from torch.fx import symbolic_trace

traced = symbolic_trace(model)
patcher = RuntimePatcher()
report = patcher.apply_patches(traced, patches=[...])
print(report.applied_count, "nodes patched")
optimized = report.module
```

---

## Custom Optimization Passes (Plugin System)

Agnitra has a ClawHub-style plugin registry. Publish your own optimization passes as Python packages:

```python
# my_package/passes.py
from agnitra.core.optimizer.pass_registry import OptimizationPass

class QuantizeInt8Pass(OptimizationPass):
    name = "quantize_int8"
    description = "Post-training int8 quantization"

    def apply(self, model, input_tensor, **kwargs):
        return torch.quantization.quantize_dynamic(model, dtype=torch.qint8)
```

```toml
# pyproject.toml
[project.entry-points."agnitra.passes"]
quantize_int8 = "my_package.passes:QuantizeInt8Pass"
```

```python
from agnitra.core.optimizer.pass_registry import PassRegistry
PassRegistry().apply("quantize_int8", model, sample)
```

---

## REST API

```bash
# Start the server
agnitra-api --host 0.0.0.0 --port 8080

# Health check
GET /health

# Synchronous optimization
POST /optimize
  -F model_graph=@graph_ir.json
  -F telemetry=@telemetry.json
  -F target=A100

# Async optimization (returns job_id immediately)
POST /optimize
  {"model_graph": [...], "telemetry": {...}, "target": "A100", "async": true}

# Poll job status
GET /jobs/{job_id}

# Real-time job status (WebSocket)
ws://host/ws/jobs/{job_id}
→ {"status": "running",   "job_id": "abc123"}
→ {"status": "completed", "job_id": "abc123", "result": {...}}

# Marketplace billing hook (AWS / GCP / Azure)
POST /usage
  {"project_id": "proj-1", "model_name": "llama3",
   "baseline": {"latency_ms": 120, "tokens_per_sec": 90},
   "optimized": {"latency_ms": 58, "tokens_per_sec": 189},
   "providers": ["aws", "gcp"]}
```

---

## Deployment

### Docker

```bash
docker build -t agnitra .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e AGNITRA_PROJECT_ID=prod \
  agnitra
```

### Kubernetes (Helm)

```bash
helm install agnitra deploy/helm/agnitra-marketplace \
  --set api.key=$AGNITRA_API_KEY \
  --set autoscaling.enabled=true
```

### AWS / GCP / Azure (Terraform)

```bash
# AWS Fargate
cd deploy/terraform/aws_marketplace && terraform apply

# Google Cloud Run
cd deploy/terraform/gcp_marketplace && terraform apply

# Azure Container Apps
cd deploy/terraform/azure_marketplace && terraform apply
```

Each module outputs a `/usage` URL ready to register with the respective cloud marketplace listing.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Developer Surface                  │
│  agnitra.optimize()  •  CLI  •  REST API  •  JS SDK  │
└──────────────────────────┬──────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────┐
│                 Runtime Optimization Agent            │
│  Fingerprint → Cache → Control Plane → LLM → RL     │
└───┬───────────────────────────────────────────┬─────┘
    │                                           │
┌───▼───────────┐                   ┌───────────▼──────┐
│  Triton Kernel │                   │  Telemetry +     │
│  Generator     │                   │  Usage Metering  │
│  + FX Patcher  │                   │  (GPU hrs, cost) │
└───────────────┘                   └──────────────────┘
                                            │
                              ┌─────────────▼────────────┐
                              │   Billing & Marketplace   │
                              │   Stripe • AWS • GCP •    │
                              │   Azure • Webhooks        │
                              └──────────────────────────┘
```

---

## Development

```bash
git clone https://github.com/agnitraai/agnitraai.git
cd agnitraai

# Install with all extras
pip install -e ".[openai,rl,nvml,marketplace]"

# Run tests
pytest -q

# Check your environment
agnitra doctor
```

Test artifacts (profiles, telemetry JSON) are written to `benchmarks/` and `agnitraai/context/`. See `.gitignore` for exclusion rules.

---

## Contributing

Contributions are welcome. Before opening a PR:

1. Run `pytest -q` — all tests must pass.
2. Run `agnitra doctor` — confirm your environment is configured.
3. Follow the coding style in [`AGENTS.md`](AGENTS.md): PEP 8, type hints, NumPy-style docstrings.
4. For new optimization passes, implement `OptimizationPass` from `agnitra.core.optimizer.pass_registry`.

See [`AGENTS.md`](AGENTS.md) for commit message format, branch naming, and PR checklist.

---

## Roadmap

- [ ] Quantization-aware optimization (INT8 / FP8 / NF4)
- [ ] Multi-GPU tensor parallelism patches
- [ ] FlashAttention-3 kernel injection
- [ ] Dashboard UI for optimization history and cost savings
- [ ] `agnitra.optimize_ctx` context manager and `@agnitra_step` decorator
- [ ] Native Hugging Face `transformers` integration
- [ ] OpenTelemetry trace export

---

## License

Apache 2.0 — free for personal and commercial use. See [`LICENSE`](LICENSE).

---

## Resources

- [PyPI](https://pypi.org/project/agnitra/) — `pip install agnitra`
- [npm](https://www.npmjs.com/package/agnitra) — `npm install agnitra`
- [OpenClaw skill](skills/agnitra/SKILL.md) — use Agnitra from WhatsApp, Slack, Telegram
- [`internal-docs/prd.md`](internal-docs/prd.md) — product roadmap and business context
- [`internal-docs/responses_api.mdx`](internal-docs/responses_api.mdx) — OpenAI Responses API spec used by the SDK
