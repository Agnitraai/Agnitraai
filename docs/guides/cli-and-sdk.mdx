---
title: "SDK & CLI Guide"
description: "Command reference and usage patterns for the Agnitra CLI and Python SDK."
---

# SDK & CLI Guide

The Agnitra CLI mirrors the Python SDK so teams can trigger optimizations, collect telemetry, and emit usage events from any environment.

## CLI Commands

### `agnitra optimize`

Optimize a TorchScript model and optionally save the optimized artifact.

```bash
agnitra optimize \
  --model tinyllama.pt \
  --input-shape 1,16,64 \
  --output dist/tinyllama_optimized.pt \
  --target A100
```

Key flags:

- `--device` — moves the model to a specific device (e.g. `cuda:0`).
- `--disable-rl` — skip PPO fine-tuning passes.
- `--offline` — disable control plane calls (requires enterprise license).
- `--require-license` — fail if license validation is unavailable.
- `--license-seat` / `--license-org` — override license metadata sent to the control plane.

### `agnitra-api`

Start the Agentic Optimization API backed by Starlette:

```bash
agnitra-api --host 0.0.0.0 --port 8080
```

Endpoints:

- `POST /optimize` — synchronous or async queued optimization.
- `GET /jobs/{id}` — poll async job status.
- `POST /usage` — convert telemetry snapshots into marketplace usage records.

API keys are read from `AGNITRA_API_KEY` (and variants) and enforced for every request.

### `agnitra-dashboard`

Spin up the HTML dashboard for local telemetry review:

```bash
agnitra-dashboard --host 127.0.0.1 --port 3000
```

## Python SDK Highlights

- `agnitra.optimize(model, input_tensor=...)` returns a `RuntimeOptimizationResult` including the optimized model, usage event, and patch metadata.
- `agnitra.sdk.resolve_input_tensor` synthesizes input tensors based on shape hints or example tensors on the module.
- Usage events expose GPU hours saved, cost savings, and marketplace metadata (see `agnitra/core/metering/usage_meter.py`).

### Example

```python
from agnitra import optimize

result = optimize(
    model,
    input_tensor=sample,
    project_id="demo-project",
    metadata={"source": "notebook"}
)

usage = result.usage_event
print(f"GPU hours saved: {usage.gpu_hours_saved:.6f}")
```

## Troubleshooting

- Missing PyTorch: install `torch>=2.0` or ensure CUDA libs are discoverable.
- Control plane unavailable: pass `--offline` or set `AGNITRA_CONTROL_PLANE_URL` to the reachable endpoint.
- Stripe/NVML optional deps: install extras (`agnitra[nvml]`, `agnitra[marketplace]`) to enable GPU telemetry and marketplace dispatchers.
