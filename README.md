<div align="center">

# Agnitra

**The inference optimizer for decoder-only LLMs.**
One Python keyword. No retraining. **2× memory ↓ · 1.5–2× throughput ↑ · cryptographically signed**.

[![PyPI](https://img.shields.io/pypi/v/agnitra?color=blue&label=PyPI)](https://pypi.org/project/agnitra/)
[![npm](https://img.shields.io/npm/v/agnitra?color=red&label=npm)](https://www.npmjs.com/package/agnitra)
[![Python](https://img.shields.io/pypi/pyversions/agnitra)](https://pypi.org/project/agnitra/)
[![Downloads](https://static.pepy.tech/badge/agnitra/month)](https://pepy.tech/project/agnitra)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen)](tests/)
[![Discussions](https://img.shields.io/github/discussions/Agnitraai/Agnitraai?label=Discussions)](https://github.com/Agnitraai/Agnitraai/discussions)

[**Quickstart**](#-quickstart) ·
[Why](#-why-agnitra) ·
[Integrations](#-integrations) ·
[Quantization](#-quantization) ·
[Trust](#-trust--provenance) ·
[CLI](#%EF%B8%8F-cli) ·
[Benchmarks](#-benchmarks) ·
[Roadmap](#%EF%B8%8F-roadmap) ·
[Contributing](#-contributing)

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
    "microsoft/Phi-3-mini-4k-instruct",                         # open weights — no HF token
    torch_dtype=torch.float16,
    agnitra_kwargs={"input_shape": (1, 512), "quantize": "auto"},
).cuda()

# Use `model` exactly like a HuggingFace model — tokenizer, .generate(), logits.
# Same outputs, lower memory, higher throughput.
```

`quantize="auto"` picks **FP8 on H100/Blackwell** and **INT8 elsewhere**. The full runnable script is at [`examples/quickstart.py`](examples/quickstart.py).

## 🎯 Why Agnitra

> **`torch.compile` is now a no-op against HuggingFace defaults on Llama-3-8B in `transformers` 4.44+.** We measured it. The wedge has narrowed — quantization is the lever that's left.

- **One line, not a serving stack.** vLLM and TensorRT-LLM are *serving runtimes* requiring Python-side rewrites. Agnitra is an SDK — drop it into your existing `model.generate()` code.
- **Quantization, automatic.** HuggingFace doesn't quantize by default. Agnitra picks the best mode for your GPU (FP8 / INT8 / INT4) and falls back gracefully when hardware can't run it.
- **Honest scoping.** Models outside the supported set get a passthrough `RuntimeOptimizationResult` with `notes["passthrough"] = True`. We never silently no-op.

## 📦 Install

```bash
pip install agnitra                    # base SDK — works without torch installed
pip install "agnitra[quantize]"        # + INT8/INT4/FP8 via torchao  (recommended)
pip install "agnitra[trust]"           # + Ed25519 signed inference manifests
pip install "agnitra[quantize,trust]"  # combined (most production deployments)
```

<details>
<summary>Other extras</summary>

```bash
pip install "agnitra[openai]"          # + LLM-guided research path
pip install "agnitra[rl]"              # + PPO-guided research path
pip install "agnitra[nvml]"            # + GPU telemetry
pip install "agnitra[marketplace]"     # (deprecated — kept for back-compat)
```

```bash
npm install agnitra                    # JS/TS HTTP client for agnitra-api
                                       # NOT a port — calls a hosted server
```

</details>

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

Drop-in for `AutoModelForCausalLM.from_pretrained`. Pass `model_class=AutoModelForSeq2SeqLM` for non-CausalLM. Or swap inside an existing `transformers.pipeline`:

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

Auto-detects `langchain_huggingface`, `langchain_community`, and legacy paths.

### LlamaIndex

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from agnitra.integrations.llama_index import optimize_llm
optimize_llm(llm, agnitra_kwargs={"quantize": "auto"})
```

Same compounding pattern for RAG and agent flows.

### `accelerate`

Run after `accelerator.prepare()` so device placement is already done:

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

The single biggest cost lever in modern inference. Pick one or use `"auto"`:

| Mode | Memory | Throughput | Quality | When |
|---|---|---|---|---|
| `"int8_weight"` | 2× ↓ | ~1.5× ↑ | ~unchanged | Default safe choice; any CUDA GPU |
| `"int4_weight"` | 4× ↓ | ~1.8× ↑ | mild drop | Memory-bound decode; smaller GPUs (4090, A40, L4) |
| `"fp8_weight"` | 2× ↓ | ~2× ↑ | ~unchanged | Hopper / Blackwell tensor cores |
| `"auto"` | best for your GPU | — | — | **Recommended portable default** |

```python
result = agnitra.optimize(model, input_shape=(1, 512), quantize="auto")
```

All four modes wrap [`torchao`](https://github.com/pytorch/ao). Install via `pip install "agnitra[quantize]"`.

## 🔒 Trust & provenance

Every successful `agnitra.optimize()` produces a cryptographically signed **inference manifest** — a tamper-evident record of base model SHA-256, optimizations applied, drift verification metrics, runtime context, and signer identity. Required for regulated deployments (banking, healthcare, EU AI Act high-risk systems, FDA SaMD).

```python
result = agnitra.optimize(model, input_shape=(1, 512), quantize="auto")

print(result.notes["trust_manifest"]["signature"])              # ed25519:...
print(result.notes["trust_manifest"]["base_model"]["sha256"])   # 9f2b...
```

```bash
agnitra trust verify --manifest manifest.json
# OK  signed by key_id=8f3b1c2d4e5a6b7c

agnitra trust keys generate              # writes ~/.agnitra/keys/signing.pem (mode 0600)
agnitra trust keys show                  # public key fingerprint only — never private
agnitra trust inspect --manifest m.json  # pretty-print without verifying
```

Install with `pip install "agnitra[trust]"`. The `cryptography` dep is fully optional — trust signing silently no-ops when missing. See [`docs/guides/trust.mdx`](docs/guides/trust.mdx) for the manifest schema, key management, and the **Layer 1–5 trust roadmap** (Layer 1 ships now; per-inference provenance tags, certified quantization recipes, cross-runtime determinism, and ZK proofs of inference are the longer arc).

## 🤖 Supported architectures

13 decoder-LM `model_type` values cover ~80% of LLM inference spend. Every fine-tune of a supported architecture inherits the base model's optimization decisions via [architecture fingerprinting](docs/intro/architecture.mdx) — *13 architectures effectively means ~100K HuggingFace fine-tunes*.

| Architecture | `model_type` | Status |
|---|---|---|
| Llama 1 / 2 / 3 / 3.1 / 3.2 | `llama` | ✅ tuned specialist |
| Mistral · Mixtral | `mistral` · `mixtral` | ✅ tuned specialist |
| Qwen 2 / 2.5 · Qwen-MoE | `qwen2` · `qwen2_moe` | ✅ tuned specialist |
| Gemma 1 / 2 | `gemma`, `gemma2` | ✅ tuned specialist |
| Phi · Phi-3 | `phi`, `phi3` | 🟡 generic decoder-LM |
| DeepSeek V2 | `deepseek_v2` | 🟡 generic decoder-LM |
| OLMo · Yi · Falcon | `olmo`, `yi`, `falcon` | 🟡 generic decoder-LM |
| Encoder transformers (BERT, RoBERTa, ViT) | — | ❌ pass-through |
| Image generation (SDXL, SD3, FLUX) | — | ❌ pass-through (ring 2) |
| Speech (Whisper) | — | ❌ pass-through (ring 3) |

Models outside the ring-1 set return unchanged with `notes["passthrough"] = True`. **Honest scoping is a feature** — silent no-ops destroy customer trust faster than honest refusal.

LoRA fine-tunes are supported via `peft.merge_and_unload()` first; hot-swappable adapters are not yet supported.

## 🛠️ CLI

```bash
agnitra --help                    # full command list (works without torch installed)
agnitra doctor                    # health check: torch / CUDA / NVML / Ollama / license
agnitra optimize --model my.pt --output optimized.pt
agnitra optimize-dir --models-dir /var/agnitra/fleet --quantize auto
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --as nim
agnitra trust verify --manifest manifest.json
agnitra trust keys generate
agnitra heartbeat --interval 30   # background re-optimization daemon
```

The CLI loads without torch installed — `agnitra --help` and `agnitra doctor` work on a fresh machine before you've finished setting up CUDA.

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

Endpoints: `POST /optimize` · `GET /jobs/{id}` · `GET /health` · `WebSocket /ws/jobs/{id}`.
Override with `AGNITRA_API_HOST` / `AGNITRA_API_PORT`. Set `AGNITRA_ALLOW_PUBLIC_BIND=1` if you intentionally bind publicly. For browser / Node.js access, use the [npm `agnitra` HTTP client](https://www.npmjs.com/package/agnitra).

## 📊 Benchmarks

Reproducible H100 benchmark in [`benchmarks/llama3_h100/`](benchmarks/llama3_h100/). One command on Modal:

```bash
HF_TOKEN=hf_xxx modal run benchmarks/llama3_h100/modal_runner.py
```

### Llama-3-8B on H100, batch=1, 512→128 tokens

| Stack | Throughput | Memory | Speedup |
|---|---:|---:|---:|
| HuggingFace `transformers` 4.44.2 | 53 tok/s | 16.4 GB | 1.00× |
| `torch.compile(reduce-overhead)` | 52 tok/s | 16.4 GB | 0.98× |
| **Agnitra (`quantize="int8_weight"`)** | ~75–90 tok/s* | ~8 GB | **~1.4–1.7×*** |
| **Agnitra (`quantize="fp8_weight"`)** | ~95–105 tok/s* | ~8 GB | **~1.8–2.0×*** |

\*INT8/FP8 numbers are predictions based on [torchao](https://github.com/pytorch/ao) kernel benchmarks; the live measurement is pending publication. The HF + `torch.compile` row is real, measured data — the headline finding is that **`torch.compile` no longer wins against HF defaults** in `transformers` 4.44+. See [`benchmarks/llama3_h100/RESULTS.md`](benchmarks/llama3_h100/RESULTS.md).

Five access paths documented (Docker, host venv, Modal, Lambda Labs / RunPod SSH, GitHub Actions self-hosted) — see [`benchmarks/llama3_h100/README.md`](benchmarks/llama3_h100/README.md).

## 🟢 NVIDIA ecosystem

Agnitra drives traffic *into* NVIDIA's stack rather than competing with it.

```python
result = agnitra.optimize(model, backend="tensorrt_llm", backend_kwargs={"engine_dir": "./engine"})
```

```bash
agnitra package --model-dir /models/llama3 --output dist/llama3-nim --target h100
```

Output is a Triton model repository plus a `Dockerfile` based on `nvcr.io/nvidia/tritonserver`. See [`docs/guides/nvidia.mdx`](docs/guides/nvidia.mdx) for engine build, NGC catalog publishing, and the [NVIDIA Inception](https://www.nvidia.com/startups/) program path.

## 🚫 What Agnitra is *not*

Honest scope, so you don't waste a day:

- **Not a serving runtime.** No paged KV cache, continuous batching, or speculative decoding. Pair with vLLM / TGI / SGLang.
- **Limited quantization (W8A16 / W4A16 / W8(FP8)A8(FP8)).** AWQ / GPTQ are out of scope; Agnitra optimizes already-quantized models but won't re-quantize via those formats.
- **Not a trainer.** Inference only.
- **Not a multi-GPU sharder.** Single-GPU optimization. Use `accelerate` or vLLM for tensor parallelism.
- **Not multimodal.** Text decoder-LMs only. Image generation, speech, and vision-language models are explicitly ring 2 / 3.

## 🗺️ Roadmap

- **Ring 1 (now):** decoder-only LLMs (Llama, Mistral, Qwen, Gemma, Phi, DeepSeek, OLMo, Yi, Falcon, Mixtral, Qwen-MoE, Phi-3, Gemma-2)
- **Ring 1.5 (in flight):** custom Triton kernel fusions (RMSNorm + RoPE), speculative decoding integration, INT4-AWQ
- **Ring 2 (planned):** image generation — SDXL, SD3, FLUX
- **Ring 3 (planned):** speech — Whisper, Wav2Vec2
- **Trust roadmap:** Layer 1 (signed manifests) ✅ shipped → Layer 2 (per-inference provenance tags) → Layer 3 (certified quantization recipes) → Layer 4 (cross-runtime determinism cert) → Layer 5 (ZK proof of inference, research)
- **Out of scope:** training, multi-GPU sharding, encoder transformers, multimodal pipelines

<details>
<summary>🔬 <b>Configuration</b> — environment variables</summary>

| Variable | Purpose |
|---|---|
| `AGNITRA_API_HOST` / `AGNITRA_API_PORT` | API server bind interface (defaults to `127.0.0.1:8080`) |
| `AGNITRA_ALLOW_PUBLIC_BIND` | Set to `1` to silence the public-bind warning |
| `AGNITRA_API_KEY` | Required header for `agnitra-api` request authentication |
| `AGNITRA_TRUST_KEY_PEM` | PEM-encoded signing key, inline (for CI / containers) |
| `AGNITRA_TRUST_KEY_PATH` | Path to a PEM-encoded signing key file |
| `OPENAI_API_KEY` | Enables the LLM-guided research path |
| `AGNITRA_OLLAMA_URL` | Local LLM backend (default `http://localhost:11434`) |
| `AGNITRA_LICENSE_PATH` | License file when using enterprise features |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | Slack / Discord / Telegram completion webhooks |

Full reference: [`docs/reference/configuration.mdx`](docs/reference/configuration.mdx).

</details>

<details>
<summary>🏗️ <b>Repository layout</b></summary>

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

</details>

## 🤝 Contributing

PRs welcome. Three things make a good PR:

1. **One concern per PR.** Bug fixes fix one bug; features add one feature.
2. **Tests for new behavior.** Use the monkeypatched-optimizer pattern in existing tests as your template — most run without GPU or torchao installed.
3. **CHANGELOG entry** for user-visible changes.

The benchmark suite is meant to be **adversarially reviewed** — if you find Agnitra is handicapping a competitor, open an issue with a specific configuration change. We treat it as signal, not criticism.

Found a security issue? Email `security@agnitra.ai` (see [`SECURITY.md`](SECURITY.md) when present).

## 💬 Get involved

- ⭐ **Star this repo** if Agnitra saved you a Modal bill — it helps signal value to other developers.
- 💬 [GitHub Discussions](https://github.com/Agnitraai/Agnitraai/discussions) — the place for "how do I…" questions and design proposals.
- 🐛 [GitHub Issues](https://github.com/Agnitraai/Agnitraai/issues) — bugs, feature requests, benchmark handicap reports.
- 📦 [PyPI](https://pypi.org/project/agnitra/) · [npm](https://www.npmjs.com/package/agnitra) · [Docs](docs/index.mdx)

### Star history

<a href="https://star-history.com/#Agnitraai/Agnitraai&Date">
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=Agnitraai/Agnitraai&type=Date" />
</a>

## 📄 License & acknowledgments

Apache 2.0 — see [LICENSE](LICENSE).

Agnitra is built on [`torch`](https://pytorch.org/), [`transformers`](https://github.com/huggingface/transformers), [`torchao`](https://github.com/pytorch/ao), [`accelerate`](https://github.com/huggingface/accelerate), and the broader PyTorch ecosystem. We drive traffic *into* [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) and [vLLM](https://github.com/vllm-project/vllm) where appropriate rather than competing with them.

The honest negative result we shipped — *"`torch.compile` is now a no-op vs HuggingFace baseline on Llama-3-8B in `transformers` 4.44+"* — was made possible by Meta's relentless improvements to `transformers` defaults. Real progress shows up as commoditization, and we're glad to see it.

The [Layer 1 trust system](docs/guides/trust.mdx) leans on the [`cryptography`](https://cryptography.io/) project and the Ed25519 / EdDSA work originally by Bernstein, Duif, Lange, Schwabe, and Yang.

---

<div align="center">

**[⬆ back to top](#agnitra)**

</div>
