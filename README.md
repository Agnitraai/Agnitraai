# agnitraai

Agnitra is an end-to-end optimization platform that wraps model tuning, telemetry, and metered billing into a single developer flow. The SDK and CLI make `agnitra.optimize(model)` feel instantaneous while the control plane meters GPU hours for a usage-based SaaS model.

## Highlights
- Unified CLI (`agnitra`) and Python SDK for profiling, tuning, and exporting optimized TorchScript artifacts.
- Runtime optimization agent couples dynamic kernel injection with usage metering (`RuntimeOptimizationAgent` + `UsageMeter`).
- Telemetry collectors, LLM-guided kernel suggestions, and RL-backed refinements.
- Usage-based SaaS pipeline that links telemetry → usage logs → Stripe metered billing with pay-per-optimization pricing.

## Installation

### From Wheel (recommended)

```

Inject a custom `UsageMeter` if you need different pricing, and inspect `result.baseline` / `result.optimized` snapshots for telemetry, GPU usage, and billing metadata that can be forwarded to your control plane.
pip install agnitra
```

Rebuild the wheel from source when iterating locally:

```
python -m build --wheel
python -m pip install --force-reinstall --no-index --find-links dist agnitra
```

### From Source

```
pip install -e .[openai,rl]
```

Optional extras:
- `agnitra[openai]` → OpenAI Responses API client.
- `agnitra[rl]` → PPO tuning via Stable Baselines3 + Gymnasium.
- `agnitra[nvml]` → GPU telemetry using NVML.
- `agnitra[marketplace]` → Cloud marketplace adapters (`boto3`, `httpx`, `google-auth`).

## Quick Start

### 1. Watch the walkthrough

- `launch_demo.mp4` – short narrated slides covering the milestone demo.

### 2. Run the milestone script

```
python demo.py --sample-shape 1,16,64
```

The script performs three sequential demos:

| Segment | What it shows |
| --- | --- |
| **Baseline vs Optimized** | Runs `RuntimeOptimizationAgent` (`agnitra.optimize`) on the TinyLlama fixture and reports latency + billing uplift. |
| **CLI Optimization** | Executes `agnitra optimize --model tinyllama.pt` and shows the pay-per-optimization summary before saving the artifact. |
| **Kernel Injection** | Generates a Triton kernel and swaps an FX node via `RuntimePatcher`. |

Each segment emits a structured usage event. The CLI mirrors the SDK output, printing tokens/sec uplift, GPU hours saved, and the metered charge so teams can verify billing before rollout.

### 3. CLI cheatsheet

```
agnitra --help
agnitra optimize --model tinyllama.pt --input-shape 1,16,64
```

### 3.5 Agentic Optimization API

1. Launch the Starlette service:
   ```
   agnitra-api --host 127.0.0.1 --port 8080
   ```
   (equivalent to `uvicorn agnitra.api.app:create_app`).
2. Call the endpoint with graph + telemetry artifacts:
   ```
   curl -X POST http://127.0.0.1:8080/optimize \
     -F model_graph=@graph_ir.json \
     -F telemetry=@telemetry.json \
     -F target=A100
   ```
   The JSON response includes an optimized IR graph, generated Triton kernel source, and FX patch instructions.
3. For JSON payloads, send `{"target": "...", "model_graph": [...], "telemetry": {...}}` with `Content-Type: application/json`.

### 4. Marketplace Usage Endpoint

The API now exposes `POST /usage`, a marketplace-compatible billing hook that
accepts baseline/optimized telemetry or a precomputed `UsageEvent`. The endpoint
returns the normalised usage payload alongside dispatch results for AWS, GCP,
and Azure marketplace adapters.

```
curl -X POST http://127.0.0.1:8080/usage \
  -H "Content-Type: application/json" \
  -d '{
        "project_id": "demo-project",
        "model_name": "tinyllama",
        "baseline": {"latency_ms": 120, "tokens_per_sec": 90, "tokens_processed": 2048},
        "optimized": {"latency_ms": 80, "tokens_per_sec": 140, "tokens_processed": 2048},
        "tokens_processed": 2048,
        "providers": ["aws", "gcp"]
      }'
```

When marketplace credentials or SDKs are not present the adapters respond with
`status: "skipped"` or `status: "deferred"` so that the control plane can retry.

### 4. SDK in your code

```python
import torch
from agnitra import optimize

model = torch.jit.load("tinyllama.pt")
sample = torch.randn(1, 16, 64)

result = optimize(
    model,
    input_tensor=sample,
    enable_rl=False,
    project_id="demo",
)
optimized = result.optimized_model
usage_event = result.usage_event
print(f"GPU hours saved: {usage_event.gpu_hours_saved:.6f}, billable: {usage_event.total_billable:.4f} {usage_event.currency}")
```

## Usage-Based SaaS Architecture

The repository tracks the implementation plan for a pay-per-optimization product. Key building blocks:

- **Runtime Agent** – `agnitra.core.runtime.agent.RuntimeOptimizationAgent` intercepts CUDA/ROCm/Triton workloads, applies runtime patches, and records tokens/sec, latency, and GPU utilisation before/after optimization.
- **Telemetry + Metering** – `UsageMeter` converts those snapshots into GPU-hour, cost-savings, and billable records that the CLI/SDK emit as structured usage events.
- **Control Plane** – FastAPI (REST) + gRPC fronting an `Optimize()` endpoint. Async workers aggregate usage, enrich with cost data, and call Stripe Metered Billing. Webhooks reconcile invoices with project owners.
- **Billing Loop** – Stripe metered usage records keyed by project ID + region + tag. Usage snapshots are bundled into invoices. Saved reports highlight cost savings vs baseline compute spend.
- **Developer Surface** – function wrapper (`agnitra.optimize`), context manager (`optimize_ctx`), and decorator (`agnitra_step`) so optimisation happens automatically.

### Repository Layout (Monorepo blueprint)

```
agnitra/
├─ sdk/python/
│  ├─ agnitra/
│  │  ├─ optimize.py        # optimize(), optimize_ctx, agnitra_step
│  │  ├─ passes/            # pluggable optimization passes
│  │  ├─ backends/          # torch/tf/jax adapters
│  │  ├─ telemetry.py       # usage events buffer + signer
│  │  ├─ auth.py            # session token management
│  │  ├─ config.py          # Config dataclass
│  │  └─ cli.py             # Click CLI wiring
│  └─ pyproject.toml
├─ control-plane/
│  ├─ api/                  # REST/gRPC services
│  ├─ metering/             # aggregation, rating, invoicing
│  ├─ billing/              # Stripe/Paddle adapters + webhooks
│  └─ db/                   # migrations for usage tables
└─ infra/                   # Docker, Helm, Terraform manifests
```

### Metering Flow

1. SDK attaches to a model – emits a `usage.attach` event with tags (`model`, `env`, `region`).
2. Runtime agent records baseline + optimized telemetry (latency, tokens/sec, GPU utilisation).
3. Control plane ingests events, aggregates GPU hours optimised, and calculates uplift.
4. Stripe metered billing rates GPU hours / tokens and issues invoices.
5. Dashboard (future) visualises savings and lets teams approve optimisations before rollout.

## Deployment & Marketplace Integration

- **Docker** – Build a containerised runtime with `docker build -t agnitra-marketplace .`
  and run it using `docker run -p 8080:8080 agnitra-marketplace`.
- **Helm** – `deploy/helm/agnitra-marketplace` packages the API for Kubernetes,
  exposing configuration for marketplace credentials, autoscaling, and ingress.
- **Terraform** – Turn-key modules exist for AWS Fargate (`deploy/terraform/aws_marketplace`),
  Google Cloud Run (`deploy/terraform/gcp_marketplace`), and Azure Container Apps
  (`deploy/terraform/azure_marketplace`). Each module outputs a ready-to-register
  `/usage` endpoint.
- **CloudFormation** – `deploy/cloudformation/aws-marketplace.yaml` offers an AWS-native
  template for rapid provisioning without Terraform.

Register the emitted `/usage` URL with the respective marketplace listing so that
usage events flow into the provider-managed billing pipeline.

## Profiling & Visualisation

The classic profiling flow remains available:

1. Profile a model: `python -m agnitra.cli profile tinyllama.pt --input-shape 1,16,64 --output telemetry.json`
2. Load telemetry + extract an FX graph IR via `agnitra.core.ir.graph_extractor`.
3. Explore results inside `agnitra_enhanced_demo.ipynb` (Colab badge included). The notebook now includes an **Agentic Optimization API (v1.0)** section that exercises `run_agentic_optimization` end-to-end and previews the patch plan + Triton kernel produced by the server.

## Development

```
pytest -q
```

Artifacts generated by tests (profiles, telemetry) live under `benchmarks/` and `agnitraai/context/`; consult `.gitignore` for the latest ignore rules. Update docs when the CLI or SDK experience changes.

## Resources
- `docs/responses_api.md` – OpenAI Responses API spec followed by the SDK.
- `docs/non_interactive_codex_usage.txt` – headless Codex automation notes.
- `docs/ui_ux_handoff.md` – UX flows and SaaS onboarding notes.
- `AGENTS.md` + `notes.yaml` – roadmap fragments and agent experiments.
