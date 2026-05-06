# Changelog

All notable changes to Agnitra are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.3] — 2026-05-06

The trust release. Adds the cryptographic provenance layer for
regulated LLM deployments. Every successful ``agnitra.optimize()``
now produces a tamper-evident, Ed25519-signed manifest of the base
model's deterministic SHA-256, the optimizations applied, the
verification metrics, the runtime context, and the signer identity.

### Added

- **`agnitra/trust/`** — new package (~700 LOC) implementing Layer 1
  of the trust roadmap. ``InferenceManifest`` schema, deterministic
  ``model_sha256`` over weights, Ed25519 keypair management,
  ``sign_manifest`` / ``verify_manifest`` helpers. The
  ``cryptography`` package is an optional dep — install via
  ``pip install "agnitra[trust]"``.
- **``agnitra.optimize``** gains ``trust_sign=True`` (default on) and
  ``trust_source: Optional[str]`` parameters. The manifest is
  automatically attached to ``result.notes["trust_manifest"]``.
- **``agnitra trust`` CLI group** with five subcommands:
  ``trust inspect``, ``trust verify``, ``trust keys generate``,
  ``trust keys show``, ``trust verify --trusted-keys``.
- ``[trust]`` optional dependency group in ``pyproject.toml``.
- ``docs/guides/trust.mdx`` — full user-facing documentation
  including the Layer 1-5 trust roadmap.

### Why this exists

Regulated industries (banking, healthcare, EU AI Act high-risk
systems, FDA SaMD) need cryptographic proof of which model was
optimized and that the result was validated. No competitor
structurally can host this — NVIDIA / vLLM / Together / HuggingFace
all have conflicts of interest as either hardware vendors, runtime
maintainers, or platform operators. An independent, foundation-
governed standard is the play; Layer 1 is the technical foundation.

## [0.2.2] — 2026-05-06

Patch release fixing 9 first-run bugs found by walking the codebase as
different developer personas. Without these fixes a fresh
``pip install agnitra`` followed by ``agnitra --help`` crashed with a
torch ImportError before any user code ran.

### Fixed

- **`import agnitra` no longer requires torch.** The package now uses
  PEP 562 ``__getattr__`` for lazy attribute access; ``__version__``
  is read eagerly, but ``optimize`` / ``optimize_model`` / the ``sdk``
  submodule are loaded only on first attribute access.
- **`agnitra --help` no longer requires torch.** CLI handlers defer
  the ``from .sdk import optimize`` import until they actually need
  it. ``agnitra doctor`` works as advertised even when torch is
  missing.
- **`pip install -e .` works on Python 3.9/3.10.** Added
  ``tomli; python_version<'3.11'`` to ``[build-system].requires`` so
  the setup.py shim's pyproject.toml read works on older Pythons.
- **`_MODE_RESOLVERS` capture-at-load bug.** PR #20's quantization
  refactor used a dict-of-functions that captured references at
  module load. Tests using ``monkeypatch.setattr(_quantization,
  "_resolve_int8_weight_only_config", ...)`` had no effect on
  dispatch. Replaced with a string-name table resolved via
  ``getattr(this_module, name)`` at call time.
- **`_DISPATCH` capture-at-load bug.** Same pattern in
  ``decoder_lm/__init__.py``. Now maps ``model_type`` to the
  specialist *module*, with ``module.optimize`` resolved at call time
  so monkeypatching ``llama.optimize`` works.
- **`detect_architecture` crash on descriptor-raising configs.**
  ``getattr(obj, "attr", default)`` only swallows ``AttributeError``;
  ``@property``-raised exceptions propagated. Wrapped
  ``_config_model_type`` in defensive try/except.
- **Tests that used `offline=True` failed without a license file.**
  Removed the flag from tests that already monkeypatch the optimizer.
- **Validation tests assumed the legacy agent path.**
  ``use_specialist=True`` (the default) bypasses the mocked
  ``RuntimeOptimizationAgent``. Tests now pass ``use_specialist=False``
  to force the legacy agent path they're actually testing.

### Changed

- ``torch>=2.0`` is now declared as a runtime dependency in
  ``pyproject.toml``. Previously the package required torch but didn't
  declare it; ``pip install agnitra`` left users to install torch
  themselves.

## [0.2.1] — 2026-05-06

Patch release shipping NVIDIA-ecosystem surface and three additional
quantization modes. Same wedge as 0.2.0; broader hardware reach.

### Added

- **TensorRT-LLM backend** — `agnitra.optimize(model, backend="tensorrt_llm",
  backend_kwargs={"engine_dir": "./engine"})` wraps a pre-built
  TensorRT-LLM engine in a HuggingFace-shaped runtime via
  `agnitra/integrations/tensorrt_llm.py`. Exposes `.generate()`
  signature unchanged so existing code paths work.
- **NIM packaging** — new `agnitra package --model-dir X --output Y --as nim`
  CLI command produces a Triton model repository plus a Dockerfile
  based on `nvcr.io/nvidia/tritonserver`. Output is a starting
  template for NIM-blessed deployment.
- **INT4 weight-only quantization** — `quantize="int4_weight"`. ~4× memory
  reduction, ~1.6-2.0× throughput on memory-bound decode. Mild quality
  drop; verify against eval set.
- **FP8 weight-only quantization** — `quantize="fp8_weight"`. Native FP8
  tensor cores on H100 / H200 / Blackwell. ~2× throughput vs FP16 with
  near-zero quality loss. Falls back to slow emulation on pre-Hopper —
  use `"auto"` to avoid that footgun.
- **Auto-mode quantization** — `quantize="auto"` inspects the GPU's
  compute capability and picks FP8 on Hopper+ / Blackwell, INT8
  elsewhere. Recommended portable default.
- **NVIDIA docs guide** — `docs/guides/nvidia.mdx` covering TensorRT-LLM
  backend, engine build, NIM packaging, Triton, NGC catalog, and
  Inception program path.

### Changed

- README — new "NVIDIA ecosystem" section, expanded quantization table
  showing all four modes (int8 / int4 / fp8 / auto).
- `apply_universal` in `agnitra/optimizers/decoder_lm/_passes.py` now
  threads any non-None `quantize` value through the unified
  `apply_quantization(model, mode)` helper. Removes the hardcoded
  INT8-only branch.

## [0.2.0] — 2026-05-06

The wedge release. Agnitra narrows from "the PyTorch inference
optimizer" to "the inference optimizer for decoder-only LLMs" and
ships the first mechanism with a real speedup hypothesis vs. the
HuggingFace baseline (INT8 weight-only quantization).

### Added

- **`agnitra/optimizers/`** — architecture detection + ring-1 routing.
  `agnitra.optimize` now identifies the model architecture (Llama,
  Mistral, Qwen, Gemma, Phi, DeepSeek, OLMo, Yi, Falcon, Mixtral,
  Qwen2-MoE, Phi-3, Gemma-2) and routes to a hard-coded specialist
  pipeline. Architectures outside the supported set get an honest
  passthrough result with `notes["passthrough"]=True`.
- **`agnitra/optimizers/decoder_lm/`** — specialist optimizers per
  architecture. Currently delegate to a shared `apply_universal`
  sequence (TF32 + SDPA verify + static KV cache + `torch.compile`).
  Architecture-specific Triton fusions (RoPE+RMSNorm, fused QKV) are
  documented as TODOs requiring GPU access.
- **INT8 weight-only quantization** via `quantize="int8_weight"`
  parameter on `agnitra.optimize`. Wraps `torchao`'s
  `int8_weight_only` config. Predicted impact on Llama-3-8B:
  ~1.3-1.7× throughput on memory-bound decode, ~2× memory reduction.
  Install via `pip install "agnitra[quantize]"`.
- **`agnitra/core/runtime/validation.py`** — output-drift safety net.
  After optimization, the SDK runs the baseline and optimized models
  on a sample input and compares cosine similarity, max-abs-diff, and
  argmax-match-rate. Drift exceeding tolerance reverts to the baseline
  with `notes["reverted_due_to_drift"]=True`. Default-on; opt-out via
  `validate=False`.
- **`agnitra/core/runtime/fingerprint.py`** — `architecture_fingerprint`
  and `architecture_signature` helpers. Two models with the same
  architecture (e.g., a fine-tune and its base) produce identical
  signatures, enabling per-architecture optimization-decision caching.
- **`agnitra.integrations.huggingface`** — `AgnitraModel.from_pretrained`,
  `wrap_model`, and `optimize_pipeline`. Drop-in replacement for
  `AutoModelForCausalLM.from_pretrained`.
- **`agnitra.integrations.accelerate_helpers.optimize_after_prepare`** —
  for users who go through `accelerate.Accelerator.prepare`.
- **`agnitra.integrations.langchain.optimize_llm`** — swaps the model
  inside a LangChain `HuggingFacePipeline` LLM in place. Agents
  downstream inherit the speedup.
- **`agnitra.integrations.llama_index.optimize_llm`** — same pattern
  for `HuggingFaceLLM`. RAG and agent flows.
- **`agnitra optimize-dir --models-dir`** CLI command — batch-optimize
  every HF model in a directory. Architecture-fingerprint cache means
  same-architecture fine-tunes after the first one optimize instantly.
- **`benchmarks/llama3_h100/`** — reproducible H100 benchmark suite
  with five customer access paths (Docker, host venv, Modal,
  Lambda/RunPod SSH, GitHub Actions self-hosted).
- **`.github/workflows/{ci,benchmark}.yml`** — pytest matrix on
  Python 3.10/3.11/3.12 plus self-hosted H100 benchmark workflow with
  >5% throughput regression gate.
- **`examples/quickstart_hf.py`**, **`quickstart_langchain.py`**,
  **`quickstart_llama_index.py`** — runnable minimal examples for
  each integration.

### Changed

- **`agnitra.optimize` signature** — now accepts `quantize`,
  `validate`, `fallback_on_regression`, `output_tolerance`,
  `argmax_match_threshold`, `use_specialist`. All default to
  conservative values; existing callers see no behavior change.
- **`agnitra-api`** server defaults — bind to `127.0.0.1:8080`
  instead of `0.0.0.0` to avoid accidental public exposure.
  `AGNITRA_API_HOST`, `AGNITRA_API_PORT`, `AGNITRA_ALLOW_PUBLIC_BIND`
  environment variables added.
- **README** — narrowed wedge sentence to "decoder-only LLMs," added
  supported-architecture table with status flags, LoRA stance, and
  roadmap rings. Length reduced from 564 to ~225 lines.

### Removed

- **`agnitra/dashboard/`** — web dashboard package and the
  `agnitra-dashboard` console_script. Off-thesis for the new wedge.
- **`agnitra/api/marketplace.py`** + the `[marketplace]` extras and
  the `/usage` API route — cloud-marketplace billing was off-thesis.
- **`agnitra/api/license_server.py`** — orphaned, no callers.
- **`js/`** — the TypeScript client was removed in this release.
  See PR #16 for a minimal HTTP client restoration if you need
  browser/Node access to the API server.
- **`skills/agnitra/`** — WhatsApp ClawHub skill, off-thesis.
- **`agnitra_enhanced_demo.ipynb`** (5,124 lines) — replaced with
  `examples/quickstart.py`.
- **`deploy/`** — Helm/Terraform/CloudFormation for cloud-marketplace
  deployments, all marketplace-only.

### Fixed

- **`agnitra/telemetry_collector.py`** — replaced inline
  `__import__('functools')` with a top-level import; passes security
  review tooling cleanly.
- **`benchmarks/llama3_h100/modal_runner.py`** — safe path resolution
  inside the Modal container (was `IndexError: 2`).
- **`benchmarks/llama3_h100/common.py`** — fixed relative-import bug
  that broke every runner at import time.
- **`benchmarks/llama3_h100/Dockerfile`** + README — corrected build
  context (must be repo root) and volume mount path so JSON outputs
  survive `docker rm`.

## [0.1.0] — 2026 (initial)

Initial release. Profiler, FX graph extractor, LLM-guided
optimization, RL-guided optimization, kernel generator, runtime
patcher, telemetry, metering, billing infrastructure.
