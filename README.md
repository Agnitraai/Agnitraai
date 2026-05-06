<div align="center">

# Agnitra

**The inference optimizer for decoder-only LLMs.**
One line of Python. No retraining. Quantization-aware. Cryptographically signed. Honest about what it does and doesn't.

[![PyPI](https://img.shields.io/pypi/v/agnitra?color=blue&label=PyPI)](https://pypi.org/project/agnitra/)
[![npm](https://img.shields.io/npm/v/agnitra?color=red&label=npm)](https://www.npmjs.com/package/agnitra)
[![Python](https://img.shields.io/pypi/pyversions/agnitra)](https://pypi.org/project/agnitra/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)](tests/)

[Quickstart](#-quickstart) ·
[Why](#-why-agnitra) ·
[Integrations](#-integrations) ·
[Quantization](#-quantization) ·
[Trust](#-trust--provenance) ·
[CLI](#%EF%B8%8F-cli) ·
[Benchmarks](#-benchmarks) ·
[Roadmap](#%EF%B8%8F-roadmap)

</div>

---

## ⚡ Quickstart

```bash
pip install "agnitra[quantize]"
```

```python
import torch
from agnitra.integrations.huggingface import AgnitraModel

model = AgnitraModel.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",      # open weights — no HF token needed
    torch_dtype=torch.float16,
    agnitra_kwargs={"input_shape": (1, 512), "quantize": "auto"},
).cuda()

# Use `model` exactly like a HuggingFace model — tokenizer, .generate(), logits.
# The optimizer only changes how fast the same outputs come out.
```

`quantize="auto"` picks FP8 on Hopper / Blackwell GPUs and INT8 elsewhere. Full runnable script: [`examples/quickstart.py`](examples/quickstart.py).

## 🎯 Why Agnitra

Three honest reasons to choose Agnitra:

1. **One line, not a serving stack.** `torch.compile` already absorbs the easy graph-level wins; vLLM and TensorRT-LLM are *serving runtimes* that require Python-side rewrites. Agnitra is an SDK — drop into your existing `model.generate()` code unchanged.
2. **Quantization is the lever, and it's automatic.** HuggingFace doesn't quantize by default. Agnitra picks the best mode for your GPU (FP8 / INT8 / INT4) and falls back gracefully on hardware that can't run it.
3. **Honest scoping.** Models outside the supported set get a passthrough `RuntimeOptimizationResult` with `notes["passthrough"] = True`. We never silently no-op. You always know whether the optimizer did anything.

## 📦 Install

```bash
pip install agnitra                    # base SDK (lazy-imports torch, no GPU required)
pip install "agnitra[quantize]"        # + INT8/INT4/FP8 via torchao  (recommended)
pip install "agnitra[trust]"           # + cryptographic manifest signing
pip install "agnitra[openai]"          # + LLM-guided research path
pip install "agnitra[rl]"              # + PPO-guided research path
pip install "agnitra[nvml]"            # + GPU telemetry
pip install "agnitra[quantize,trust]"  # combined (most production deployments)
```

```bash
npm install agnitra                    # JS/TS HTTP client for agnitra-api
                                       # NOT a port — calls a hosted server
```

## 🔌 Integrations

Five drop-in entry points — same wedge across every popular LLM framework.

### HuggingFace `transformers`

```python
from agnitra.integrations.huggingface import AgnitraModel

model = AgnitraModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype=torch.float16,
    agnitra_kwargs={"input_shape": (1, 512), "quantize": "auto"},
).cuda()
```

Drop-in for `AutoModelForCausalLM.from_pretrained`. For non-CausalLM workloads pass `model_class=AutoModelForSeq2SeqLM` (or any other `AutoModelFor...` class).

For an existing pipeline, swap the inner model in place:

```python
from agnitra.integrations.huggingface import optimize_pipeline
optimize_pipeline(pipe, agnitra_kwargs={"input_shape": (1, 512)})
```

### LangChain

Agents call the LLM many times per task — model speedups compound into pipeline speedups.

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

Same compounding pattern for RAG and agent flows.

### `accelerate`

Run after `accelerator.prepare()` so device placement and distributed wrapping are already done:

```python
from accelerate import Accelerator
from agnitra.integrations.accelerate_helpers import optimize_after_prepare

accelerator = Accelerator()
model = accelerator.prepare(model)
model = optimize_after_prepare(model, input_shape=(1, 512))
```

### NVIDIA TensorRT-LLM

Wraps a pre-built TensorRT-LLM engine in a HuggingFace-shaped runtime:

```python
result = agnitra.optimize(
    model,
    backend="tensorrt_llm",
    backend_kwargs={"engine_dir": "./engine"},
)
```

See [`docs/guides/nvidia.mdx`](docs/guides/nvidia.mdx) for engine build, NIM packaging, and the NVIDIA Inception path.

## 🔧 Quantization

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

All four modes wrap [`torchao`](https://github.com/pytorch/ao). Install via `pip install "agnitra[quantize]"`.

## 🔒 Trust & provenance

Every successful `agnitra.optimize()` call now produces a cryptographically signed **inference manifest** — a tamper-evident record of:

- the base model's deterministic SHA-256 over weights
- the architecture fingerprint (the cross-fine-tune cache key)
- every optimization step that ran
- post-optimization drift verification metrics
- runtime context (torch / CUDA / hardware UUID / agnitra version)
- the signer's public key + key fingerprint
- an Ed25519 signature over the canonical bytes of all of the above

Required for regulated deployments (banking, healthcare, EU AI Act high-risk systems, FDA SaMD).

```python
result = agnitra.optimize(model, input_shape=(1, 512), quantize="auto")
print(result.notes["trust_manifest"]["signature"])
# ed25519:...
print(result.notes["trust_manifest"]["base_model"]["sha256"])
# 9f2b...
```

CLI verification:

```bash
agnitra trust verify --manifest manifest.json
# OK  signed by key_id=8f3b1c2d4e5a6b7c

agnitra trust keys generate              # writes ~/.agnitra/keys/signing.pem
agnitra trust keys show                  # prints the public key fingerprint
agnitra trust inspect --manifest m.json  # pretty-print without verifying
```

Install with `pip install "agnitra[trust]"`. See [`docs/guides/trust.mdx`](docs/guides/trust.mdx) for the manifest schema, key management, and the layered roadmap (Layer 1 ships now; per-inference provenance tags, certified quantization recipes, and ZK proofs of inference are the longer arc).

## 🤖 Supported architectures

13 decoder-LM `model_type` values cover ~80% of LLM inference spend. Every fine-tune of a supported architecture inherits the base model's optimization decisions via [architecture fingerprinting](docs/intro/architecture.mdx) — *13 architectures effectively means ~100K HuggingFace fine-tunes*.

| Architecture | `model_type` | Status |
|---|---|---|
| Llama 1 / 2 / 3 / 3.1 / 3.2 | `llama` | ✅ tuned specialist |
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

Models outside the supported set get a passthrough `RuntimeOptimizationResult` with `notes["passthrough"] = True`. **Honest scoping is a feature** — silent no-ops destroy customer trust faster than honest refusal.

LoRA fine-tunes are supported via `peft.merge_and_unload()` first; hot-swappable adapters are not yet supported.

## 🛠️ CLI

```bash
agnitra optimize --model my_model.pt --output optimized.pt
agnitra optimize-dir --models-dir /var/agnitra/fleet --quantize auto
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --as nim
agnitra trust verify --manifest manifest.json
agnitra trust keys generate
agnitra doctor                  # health check: torch / CUDA / NVML / Ollama / license
agnitra heartbeat --interval 30 # background re-optimization daemon
agnitra --help                  # full command list
```

The CLI loads without torch installed — `agnitra --help` and `agnitra doctor` work on a fresh machine before you've set up CUDA. Heavy commands (`optimize`, `optimize-dir`) lazy-import torch only when invoked.

### Fine-tune farms (`agnitra optimize-dir`)

The killer feature for production fleets running 50+ Llama-3 fine-tunes per customer. The architecture-fingerprint cache reuses optimization decisions across same-architecture variants:

```bash
agnitra optimize-dir --models-dir /var/agnitra/fleet --quantize auto
# Optimizing customer-A-llama3 ...    (8 minutes — real work)
# Optimizing customer-B-llama3 ...    cache hit (same architecture as customer-A) — instant
# Optimizing customer-C-llama3 ...    cache hit — instant
# ... 47 more fine-tunes ...          all cache hits — instant
```

## 🌐 API server (optional)

```bash
agnitra-api    # binds to 127.0.0.1:8080 by default
```

Endpoints:
- `POST /optimize` — submit an optimization job
- `GET /jobs/{id}` — poll status
- `GET /health` — liveness probe
- `WebSocket /ws/jobs/{id}` — real-time job updates

Override the bind interface with `AGNITRA_API_HOST` / `AGNITRA_API_PORT`. Set `AGNITRA_ALLOW_PUBLIC_BIND=1` if you intentionally bind to a public interface. For browser / Node.js access, use the [npm `agnitra` HTTP client](https://www.npmjs.com/package/agnitra).

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

*INT8/FP8 numbers are predictions based on torchao kernel benchmarks; the live measurement is pending publication. The HF + `torch.compile` row is real, measured data — the headline finding is that **`torch.compile` no longer wins against HF defaults** in `transformers` 4.44+. See [`benchmarks/llama3_h100/RESULTS.md`](benchmarks/llama3_h100/RESULTS.md).

Five access paths documented for the benchmark suite (Docker, host venv, Modal, Lambda Labs / RunPod SSH, GitHub Actions self-hosted) — see [`benchmarks/llama3_h100/README.md`](benchmarks/llama3_h100/README.md).

## 🟢 NVIDIA ecosystem

Agnitra drives traffic *into* NVIDIA's stack rather than competing with it. Most HuggingFace developers cannot use TensorRT-LLM directly because it requires C++ and a multi-step engine build. Agnitra wraps that complexity behind one keyword:

```python
result = agnitra.optimize(model, backend="tensorrt_llm", backend_kwargs={"engine_dir": "./engine"})
```

For deployable containers, package an Agnitra-optimized model as a NIM-compatible Triton bundle:

```bash
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --target h100
```

Output is a Triton model repository plus a `Dockerfile` based on `nvcr.io/nvidia/tritonserver`. See [`docs/guides/nvidia.mdx`](docs/guides/nvidia.mdx) for engine build, NGC catalog publishing, and the [NVIDIA Inception](https://www.nvidia.com/startups/) program path.

## 🚫 What Agnitra is *not*

Honest scope, so you don't waste a day:

- **Not a serving runtime.** No paged KV cache, continuous batching, or speculative decoding. Pair with vLLM / TGI / SGLang for those.
- **Limited quantization (W8A16 / W4A16 / W8(FP8)A8(FP8)).** Agnitra wraps `torchao` for these modes. INT4 / activation quantization via AWQ / GPTQ are out of scope; if you have a model already quantized via those, Agnitra will optimize it but won't re-quantize.
- **Not a trainer.** Inference only. Training-time optimization is out of scope.
- **Not a multi-GPU sharder.** Single-GPU optimization. Tensor parallelism is a separate problem (use `accelerate` or vLLM for it).
- **Not multimodal.** Text decoder-LMs only. Image generation, speech, and vision-language models are explicitly ring 2 / 3.

## 🗺️ Roadmap

- **Ring 1 (now):** decoder-only LLMs (Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, OLMo, Yi, Falcon, Mixtral, Qwen-MoE, Phi-3, Gemma-2)
- **Ring 1.5 (in flight):** custom Triton kernel fusions (RMSNorm + RoPE), speculative decoding integration, INT4-AWQ
- **Ring 2 (planned):** image generation — SDXL, SD3, FLUX
- **Ring 3 (planned):** speech — Whisper, Wav2Vec2
- **Trust roadmap:** Layer 1 (signed manifests) ✅ shipped → Layer 2 (per-inference provenance tags) → Layer 3 (certified quantization recipes) → Layer 4 (cross-runtime determinism cert) → Layer 5 (ZK proof of inference, research)
- **Out of scope:** training, multi-GPU sharding, encoder transformers, multimodal pipelines

## 🔬 Configuration

| Environment variable | Purpose |
|---|---|
| `AGNITRA_API_HOST` / `AGNITRA_API_PORT` | API server bind interface (defaults to 127.0.0.1:8080) |
| `AGNITRA_ALLOW_PUBLIC_BIND` | Set to `1` to silence the public-bind warning |
| `AGNITRA_API_KEY` | Required header for `agnitra-api` request authentication |
| `AGNITRA_TRUST_KEY_PEM` | PEM-encoded signing key, inline (for CI / containers) |
| `AGNITRA_TRUST_KEY_PATH` | Path to a PEM-encoded signing key file |
| `OPENAI_API_KEY` | Enables the LLM-guided research path |
| `AGNITRA_OLLAMA_URL` | Local LLM backend (default `http://localhost:11434`) |
| `AGNITRA_LICENSE_PATH` | License file when using enterprise features |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | Slack / Discord / Telegram webhook for completion notifications |

Full reference: [`docs/reference/configuration.mdx`](docs/reference/configuration.mdx).

## 🏗️ Repository layout

```
agnitra/
  __init__.py             lazy-loads sdk; `import agnitra` works without torch
  sdk.py                  public optimize() entry point
  cli.py                  Click CLI — optimize / optimize-dir / package / trust / doctor
  optimizers/             architecture detection + ring-1 routing
    detection.py          model_type detection (config + structural fingerprint)
    registry.py           SUPPORTED_DECODER_LM_TYPES
    decoder_lm/           llama / mistral / qwen2 / gemma specialists
      _passes.py          TF32 / SDPA / static cache / torch.compile
      _quantization.py    INT8 / INT4 / FP8 / auto via torchao
  trust/                  signed inference manifests (Layer 1)
    manifest.py           InferenceManifest schema + canonical bytes
    digest.py             deterministic model_sha256
    keys.py               Ed25519 keypair management
    sign.py / verify.py   Ed25519 signature lifecycle
  integrations/           huggingface / accelerate / langchain / llama_index / tensorrt_llm
  _sdk/                   low-level optimizer (FX, kernels, RL — research path)
  core/
    runtime/              fingerprint, validation, cache, control plane
    kernel/               Triton kernel generation
    metering/, billing/   usage events, Stripe integration
    licensing/            license validation
    notifications/        webhook notifiers
  api/                    Starlette REST API server (POST /optimize, /jobs, /ws/jobs)
benchmarks/llama3_h100/   reproducible H100 benchmark (5 access paths)
examples/                 minimal runnable scripts (HF, LangChain, LlamaIndex, CPU)
js/                       TypeScript HTTP client (npm)
docs/                     Mintlify documentation site
tests/                    111 tests; runs without GPU
```

## 🤝 Contributing

PRs welcome. Three things make a good PR:

1. **One concern per PR.** Bug fixes fix one bug; features add one feature. Mixed PRs are hard to review and harder to revert.
2. **Tests for new behavior.** Use the monkeypatched-optimizer pattern in existing tests as your template — most tests run without GPU or torchao installed.
3. **CHANGELOG entry.** If your change is user-visible, add a bullet under the appropriate `## [Unreleased]` section.

The benchmark suite is meant to be **adversarially reviewed** — if you find Agnitra is handicapping a competitor (vLLM, TensorRT-LLM, etc.), open an issue or PR with a specific configuration change. We treat this as signal, not criticism.

Found a security issue? See [SECURITY.md](SECURITY.md) (if present, otherwise email `security@agnitra.ai`).

## 📄 License

[Apache 2.0](LICENSE).

## 🙏 Acknowledgments

Agnitra is built on top of [`torch`](https://pytorch.org/), [`transformers`](https://github.com/huggingface/transformers), [`torchao`](https://github.com/pytorch/ao), [`accelerate`](https://github.com/huggingface/accelerate), and the broader PyTorch ecosystem. We drive traffic *into* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) where appropriate rather than competing with them.

The honest negative result we shipped — *"`torch.compile` is now a no-op vs HuggingFace baseline on Llama-3-8B in `transformers` 4.44+"* — was made possible by Meta's relentless improvements to `transformers` defaults. Real progress shows up as commoditization, and we're glad to see it.

The cryptographic trust layer ([Layer 1](docs/guides/trust.mdx)) leans on the [`cryptography`](https://cryptography.io/) project and the Ed25519 / EdDSA work originally by Bernstein, Duif, Lange, Schwabe, and Yang.

---

<div align="center">

**Star us if Agnitra saved you a Modal bill.**

[GitHub](https://github.com/Agnitraai/Agnitraai) · [PyPI](https://pypi.org/project/agnitra/) · [npm](https://www.npmjs.com/package/agnitra) · [Docs](docs/index.mdx) · [Benchmarks](benchmarks/llama3_h100/)

</div>
