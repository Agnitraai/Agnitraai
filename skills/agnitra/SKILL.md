# Agnitra GPU Optimizer Skill

This skill lets OpenClaw users trigger Agnitra model optimization directly from
WhatsApp, Telegram, Slack, Discord, or any other connected channel.

## What it does

Agnitra optimizes PyTorch models for faster GPU inference using LLM-guided kernel
tuning and RL-based refinement. This skill wraps the `agnitra` CLI so your AI
agent can run optimizations on demand and report GPU hours saved and latency
improvements back to your chat.

## Prerequisites

- Python ≥ 3.8 installed on the OpenClaw host
- Agnitra installed: `pip install agnitra` (or `pip install agnitra[openai,rl]`)
- A `.pt` or `.pth` TorchScript model accessible from the host

Optional but recommended:
- `OPENAI_API_KEY` set for LLM-guided suggestions
- Ollama running locally for zero-cost local LLM inference

## Installation

```bash
pip install agnitra
```

Or with all optimization extras:

```bash
pip install "agnitra[openai,rl,nvml]"
```

## Trigger phrases

The agent will invoke this skill when the user says something like:

- "optimize my model"
- "run agnitra on tinyllama.pt"
- "profile and tune my PyTorch model"
- "check agnitra health"
- "show agnitra doctor output"
- "start the optimization heartbeat"

## Actions

### optimize — Optimize a model file

```bash
agnitra optimize --model <path/to/model.pt> --input-shape <shape>
```

**Example invocations:**

```bash
# Optimize with a custom input shape
agnitra optimize --model tinyllama.pt --input-shape 1,16,64

# Optimize without RL stage (faster, good for a quick check)
agnitra optimize --model resnet50.pt --input-shape 1,3,224,224 --disable-rl

# Optimize targeting a specific GPU
agnitra optimize --model bert.pt --input-shape 1,128 --target A100

# Offline mode (requires enterprise license)
agnitra optimize --model model.pt --input-shape 1,64 --offline
```

**Output reported back to chat:**

```
Optimized model written to tinyllama_optimized.pt
Performance uplift: 18.4% | GPU hours saved: 0.000028 | Billable: 0.0001 USD
```

### doctor — Health check

```bash
agnitra doctor
```

Checks PyTorch, CUDA, pynvml, OpenAI API key, Ollama, and license configuration.
Returns a pass/fail summary to the chat.

```bash
# Also check if the local API server is running
agnitra doctor --check-api --api-url http://127.0.0.1:8080
```

### heartbeat — Start the re-optimization scheduler

```bash
# Run one cycle immediately
agnitra heartbeat --once

# Start continuous heartbeat every 30 minutes (runs in background)
agnitra heartbeat --interval 30
```

### profile — Profile telemetry only

```bash
python -m agnitra.cli profile <model.pt> --input-shape <shape> --output telemetry.json
```

## Agent instructions

When asked to optimize a model:

1. Ask the user for the model file path and input shape if not provided.
2. Run `agnitra optimize --model <path> --input-shape <shape>`.
3. Report the performance uplift, GPU hours saved, and output path back to the user.
4. If the user wants to know the system is healthy, run `agnitra doctor` first.

When asked to check the environment:

1. Run `agnitra doctor` and summarize the results.
2. If any checks fail, suggest the fix from the doctor output.

When the user says "start background optimization" or "schedule re-optimization":

1. Run `agnitra heartbeat --interval 30` in the background.
2. Confirm the heartbeat is running and explain it will re-optimize every 30 minutes.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `AGNITRA_PROJECT_ID` | `default` | Project identifier for usage tracking |
| `OPENAI_API_KEY` | — | OpenAI API key for LLM-guided optimization |
| `AGNITRA_LLM_BACKEND` | `responses` | LLM backend: `responses`, `ollama`, `codex_cli`, `auto` |
| `AGNITRA_OLLAMA_URL` | `http://localhost:11434` | Ollama server URL for local LLM |
| `AGNITRA_OLLAMA_MODEL` | `llama3` | Ollama model name |
| `AGNITRA_NOTIFY_WEBHOOK_URL` | — | Webhook URL for Slack/Discord/Telegram notifications |
| `AGNITRA_NOTIFY_CHANNEL` | `slack` | Notification channel format |
| `AGNITRA_LICENSE_PATH` | — | Path to enterprise license file |

## Example conversation

**User:** optimize my model at /models/resnet50.pt with shape 1,3,224,224

**Agent:**
```
Running: agnitra optimize --model /models/resnet50.pt --input-shape 1,3,224,224
...
Optimized model written to /models/resnet50_optimized.pt
Performance uplift: 22.1% | GPU hours saved: 0.000031 | Billable: 0.0001 USD
```

**User:** is agnitra healthy?

**Agent:**
```
Running: agnitra doctor
[OK] PyTorch installed — v2.3.0
[OK] CUDA available — 1 device(s)
[OK] pynvml (GPU telemetry)
[OK] OPENAI_API_KEY — set (length 51)
[FAIL] Ollama (local LLM) — not running
All checks passed (1 optional item not configured).
```

## Links

- PyPI: https://pypi.org/project/agnitra/
- GitHub: https://github.com/agnitraai/agnitraai
- Docs: https://github.com/agnitraai/agnitraai#readme
- Issues: https://github.com/agnitraai/agnitraai/issues
