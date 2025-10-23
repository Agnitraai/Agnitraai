---
title: "Getting Started"
description: "Install the Agnitra SDK, run the CLI, and generate optimization telemetry."
---

# Getting Started

This quickstart walks through installing the SDK, running the CLI, and inspecting optimization telemetry.

## 1. Install the SDK

Install from PyPI (recommended):

```bash
pip install agnitra
```

For editable installs while developing locally:

```bash
pip install -e .[openai,rl]
```

Optional extras:

- `agnitra[openai]` – OpenAI Responses API client bindings.
- `agnitra[rl]` – Stable Baselines3 + Gymnasium reinforcement learning add-ons.
- `agnitra[nvml]` – GPU telemetry via NVIDIA NVML.
- `agnitra[marketplace]` – Cloud marketplace adapters (`boto3`, `httpx`, `google-auth`).

## 2. Optimize a Model

```bash
agnitra optimize --model tinyllama.pt --input-shape 1,16,64
```

The CLI loads the model, generates an optimized artifact, and prints a billing snapshot (performance uplift, GPU hours saved, billable total). Pass `--output` to control the destination path.

From Python:

```python
import torch
from agnitra import optimize

model = torch.jit.load("tinyllama.pt")
sample = torch.randn(1, 16, 64)

result = optimize(model, input_tensor=sample, project_id="demo")
print(result.usage_event.total_billable)
```

## 3. Explore the API Surface

- Launch the Starlette service: `agnitra-api --host 127.0.0.1 --port 8080`
- POST graph + telemetry payloads to `/optimize` for automatic kernel suggestions.
- Forward usage events to `/usage` to dispatch marketplace billing records.

Refer to [responses_api.md](responses_api.md) for the JSON contract and webhook semantics.

## 4. Next Steps

- Review [CLI and SDK guide](cli-guide.md) for advanced flags, licensing, and offline mode.
- Follow the [Publishing Packages](publishing.md) checklist to ship PyPI and npm releases.
- Dive into [Monetization Flows](monetization_flows.md) to understand the usage-to-billing pipeline.
