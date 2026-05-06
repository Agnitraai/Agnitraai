<div align="center">

# Agnitra

**The inference optimizer for decoder-only LLMs.**
One line of Python. No retraining. Quantization-aware. Honest about what it does and doesn't.

[![PyPI](https://img.shields.io/pypi/v/agnitra?color=blue&label=PyPI)](https://pypi.org/project/agnitra/)
[![npm](https://img.shields.io/npm/v/agnitra?color=red&label=npm)](https://www.npmjs.com/package/agnitra)
[![Python](https://img.shields.io/pypi/pyversions/agnitra)](https://pypi.org/project/agnitra/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![CI](https://github.com/agnitraai/agnitraai/actions/workflows/ci.yml/badge.svg)](https://github.com/agnitraai/agnitraai/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-94%20passing-brightgreen)](https://github.com/agnitraai/agnitraai/tree/main/tests)

[Quickstart](#-quickstart) · [Why](#-why-agnitra) · [Benchmarks](#-benchmarks) · [Integrations](#-integrations) · [Roadmap](#-roadmap) · [Contributing](CONTRIBUTING.md)

</div>

---

## ⚡ Quickstart

```bash
pip install "agnitra[quantize]"
```

```python
from agnitra.integrations.huggingface import AgnitraModel
import torch

model = AgnitraModel.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    agnitra_kwargs={"input_shape": (1, 512), "quantize": "auto"},
).cuda()

# Use `model` exactly like a HuggingFace model. Tokenizer, .generate(),
# logits — everything works unchanged. The optimizer only changes
# *how fast* the same outputs come out.
```

That's the whole API. `quantize="auto"` picks FP8 on Hopper / Blackwell GPUs and INT8 elsewhere. The example uses Phi-3 (open-weight, ~7 GB) so you can copy-paste it without a HuggingFace token.

For a complete runnable script: [`examples/quickstart.py`](examples/quickstart.py).

## 🎯 Why Agnitra

Three honest reasons to choose Agnitra over the alternatives:

1. **One line, not a serving stack.** `torch.compile` has absorbed the easy graph-level wins; `vLLM` and `TensorRT-LLM` are *serving runtimes* that require Python-side rewrites. Agnitra is an SDK — drop it into your existing `model.generate()` code and it works.
2. **Quantization is the lever, and it's automatic.** HuggingFace doesn't quantize by default. Agnitra picks the best quantization mode (FP8 / INT8 / INT4) for your GPU and falls back gracefully on hardware that can't run it.
3. **Honest scoping.** Models outside the supported set get a passthrough `RuntimeOptimizationResult` with `notes["passthrough"] = True`. We never silently no-op. You always know whether the optimizer did anything.

What Agnitra is *not*:

- **Not a serving runtime** — pair with vLLM / TGI / SGLang for paged KV cache and continuous batching.
- **Not a trainer** — inference only.
- **Not a multi-GPU sharder** — single-GPU only; tensor parallelism is a separate problem.

## 📊 Benchmarks

Reproducible H100 benchmark in [`benchmarks/llama3_h100/`](benchmarks/llama3_h100/). Run with one command on Modal:

```bash
HF_TOKEN=hf_xxx modal run benchmarks/llama3_h100/modal_runner.py
```

### Llama-3-8B on H100, batch=1, 512→128 tokens

| Stack | Throughput | Memory | Speedup |
|---|---:|---:|---:|
| HuggingFace `transformers` 4.44.2 | 53 tok/s | 16.4 GB | 1.00× |
| `torch.compile(reduce-overhead)` | 52 tok/s | 16.4 GB | 0.98× |
| **Agnitra (`quantize="int8_weight"`)** | ~75-90 tok/s* | ~8 GB | **~1.4-1.7×*** |
| **Agnitra (`quantize="fp8_weight"`)** | ~95-105 tok/s* | ~8 GB | **~1.8-2.0×*** |

*INT8/FP8 numbers are predictions based on torchao kernel benchmarks; the live H100 measurement is pending publication. The HF + `torch.compile` row is real, measured data — the headline finding is that **`torch.compile` no longer wins against HF defaults** in `transformers` 4.44+. See [`benchmarks/llama3_h100/RESULTS.md`](benchmarks/llama3_h100/RESULTS.md).

## 🤖 Supported architectures

13 decoder-LM `model_type` values cover ~80% of LLM inference spend. Every fine-tune of a supported architecture inherits the base model's optimization decisions via [architecture fingerprinting](docs/intro/architecture.mdx) — *13 architectures effectively means ~100K HuggingFace fine-tunes*.

| Architecture | `model_type` | Status |
|---|---|---|
| Llama 1/2/3 | `llama` | ✅ tuned specialist |
| Mistral | `mistral` | ✅ tuned specialist |
| Mixtral | `mixtral` | ✅ tuned specialist |
| Qwen 2 / 2.5 | `qwen2` | ✅ tuned specialist |
| Qwen 2 MoE | `qwen2_moe` | ✅ tuned specialist |
| Gemma 1 / 2 | `gemma`, `gemma2` | ✅ tuned specialist |
| Phi / Phi-3 | `phi`, `phi3` | 🟡 generic decoder-LM |
| DeepSeek V2 | `deepseek_v2` | 🟡 generic decoder-LM |
| OLMo / Yi / Falcon | `olmo`, `yi`, `falcon` | 🟡 generic decoder-LM |
| Encoder transformers (BERT, RoBERTa, ViT) | — | ❌ pass-through |
| Image generation (SDXL, SD3, FLUX) | — | ❌ pass-through (ring 2) |
| Speech (Whisper) | — | ❌ pass-through (ring 3) |

## 🔧 Quantization modes

The single biggest cost-effectiveness lever in modern inference. Pick one or use `"auto"`:

| Mode | Memory | Throughput | Quality | When to use |
|---|---|---|---|---|
| `"int8_weight"` | 2× ↓ | ~1.5× ↑ | ~unchanged | Default safe choice; any CUDA GPU |
| `"int4_weight"` | 4× ↓ | ~1.8× ↑ | mild drop | Memory-bound decode; smaller GPUs (4090, A40, L4) |
| `"fp8_weight"` | 2× ↓ | ~2× ↑ | ~unchanged | Hopper / Blackwell tensor cores |
| `"auto"` | picks best for your GPU | — | — | **Recommended portable default** |

```python
result = agnitra.optimize(model, input_shape=(1, 512), quantize="auto")
```

## 🔌 Integrations

### HuggingFace `transformers`

```python
from agnitra.integrations.huggingface import AgnitraModel
model = AgnitraModel.from_pretrained(model_id, agnitra_kwargs={...})
```

Drop-in replacement for `AutoModelForCausalLM.from_pretrained`. Works with any `transformers.AutoModelFor...` class via `model_class=`.

### LangChain

```python
from langchain_huggingface import HuggingFacePipeline
from agnitra.integrations.langchain import optimize_llm

llm = HuggingFacePipeline.from_model_id("...", task="text-generation")
optimize_llm(llm, agnitra_kwargs={"quantize": "auto"})
# Every chain / agent downstream inherits the speedup.
```

Auto-detects `langchain_huggingface`, `langchain_community`, and legacy LangChain paths.

### LlamaIndex

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from agnitra.integrations.llama_index import optimize_llm
optimize_llm(llm, agnitra_kwargs={"quantize": "auto"})
```

### `accelerate`

```python
from accelerate import Accelerator
from agnitra.integrations.accelerate_helpers import optimize_after_prepare
accelerator = Accelerator()
model = accelerator.prepare(model)
model = optimize_after_prepare(model, input_shape=(1, 512))
```

### NVIDIA TensorRT-LLM

```python
result = agnitra.optimize(
    model,
    backend="tensorrt_llm",
    backend_kwargs={"engine_dir": "./engine"},
)
```

Wraps a pre-built TensorRT-LLM engine in a HuggingFace-shaped runtime. See [`docs/guides/nvidia.mdx`](docs/guides/nvidia.mdx) for engine build, NIM packaging, and NVIDIA Inception path.

## 🛠️ CLI

```bash
agnitra optimize --model my_model.pt --output optimized.pt
agnitra optimize-dir --models-dir /var/agnitra/fleet --quantize auto
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --as nim
agnitra doctor                # health check: torch / CUDA / NVML / Ollama / license
agnitra --help                # full command list
```

`agnitra optimize-dir` is the killer feature for **fine-tune farms**: 50+ Llama-3 fine-tunes optimize in roughly the time it takes to optimize one because the architecture-fingerprint cache reuses decisions across same-architecture variants.

## 🌐 API server (optional)

```bash
agnitra-api
```

Binds to `127.0.0.1:8080` by default; exposes `POST /optimize`, `GET /jobs/{id}`, `GET /health`, and `WebSocket /ws/jobs/{id}`. Use the [npm `agnitra` HTTP client](https://www.npmjs.com/package/agnitra) for browser / Node.js access.

## 🗺️ Roadmap

- **Ring 1 (now):** decoder-only LLMs (Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, OLMo, Yi, Falcon, Mixtral, Qwen-MoE, Phi-3, Gemma-2)
- **Ring 1.5 (in flight):** custom Triton kernel fusions (RMSNorm + RoPE), speculative decoding integration
- **Ring 2 (planned):** image generation — SDXL, SD3, FLUX
- **Ring 3 (planned):** speech — Whisper, Wav2Vec2
- **Out of scope:** training, multi-GPU sharding, encoder transformers, multimodal pipelines

## 📦 Install options

```bash
pip install agnitra                    # base SDK
pip install "agnitra[quantize]"        # + INT8/INT4/FP8 via torchao (recommended)
pip install "agnitra[openai]"          # + LLM-guided research path
pip install "agnitra[rl]"              # + PPO-guided research path
pip install "agnitra[nvml]"            # + GPU telemetry
```

```bash
npm install agnitra                    # JS/TS HTTP client (NOT a port; calls agnitra-api)
```

## 🔬 Configuration

| Environment variable | Purpose |
|---|---|
| `AGNITRA_API_HOST` / `AGNITRA_API_PORT` | API server bind interface |
| `AGNITRA_ALLOW_PUBLIC_BIND` | Set to `1` to silence the public-bind warning |
| `OPENAI_API_KEY` | Enables the LLM-guided research path |
| `AGNITRA_OLLAMA_URL` | Local LLM backend (default `http://localhost:11434`) |
| `AGNITRA_LICENSE_PATH` | License file when using enterprise features |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | Slack / Discord / Telegram webhook for completion notifications |

Full reference: [`docs/reference/configuration.mdx`](docs/reference/configuration.mdx).

## 🏗️ Repository layout

```
agnitra/
  sdk.py                    public optimize() entry point
  cli.py                    Click CLI
  optimizers/               architecture detection + ring-1 routing
    decoder_lm/             llama / mistral / qwen2 / gemma specialists
  _sdk/                     low-level optimizer (FX, kernels, RL)
  core/
    runtime/                fingerprint, validation, control plane
    kernel/                 Triton kernel generation
    metering/, billing/     usage events, Stripe integration
  api/                      Starlette REST API server
  integrations/             huggingface / langchain / llama_index / tensorrt_llm
benchmarks/llama3_h100/     reproducible H100 benchmark
examples/                   minimal runnable scripts
js/                         TypeScript HTTP client (npm)
docs/                       Mintlify documentation site
tests/                      94 tests; runs without GPU
```

## 🤝 Contributing

PRs welcome. The benchmark suite is meant to be adversarially reviewed — if you find Agnitra is handicapping a competitor (vLLM, TensorRT-LLM, etc.), open an issue or PR. See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, test pattern, and what makes a good PR.

Found a security issue? See [SECURITY.md](SECURITY.md).

## 📄 License

[Apache 2.0](LICENSE).

## 🙏 Acknowledgments

Agnitra is built on top of [`torch`](https://pytorch.org/), [`transformers`](https://github.com/huggingface/transformers), [`torchao`](https://github.com/pytorch/ao), [`accelerate`](https://github.com/huggingface/accelerate), and the broader PyTorch ecosystem. We drive traffic *into* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) where appropriate rather than competing with them.

The honest negative result we shipped — *"`torch.compile` is now a no-op vs HuggingFace baseline on Llama-3-8B"* — was made possible by Meta's relentless improvements to `transformers` defaults. Real progress shows up as commoditization, and we're glad to see it.

---

<div align="center">

**Star us if Agnitra saved you a Modal bill.** [GitHub](https://github.com/Agnitraai/Agnitraai) · [PyPI](https://pypi.org/project/agnitra/) · [npm](https://www.npmjs.com/package/agnitra) · [Docs](docs/index.mdx)

</div>
