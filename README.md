# agnitraai

Agnitra is an experimentation toolkit for accelerating PyTorch models with telemetry-guided optimization. It ships unified CLI and Python APIs for profiling,
FX graph introspection, and LLM/RL-assisted tuning loops.

## Features
- CLI entry points (`agnitra`, `agnitra-optimize`) to profile models and apply optimization recipes.
- Telemetry collectors that turn profiler traces into FX graphs enriched with runtime metadata.
- LLM and PPO-based optimizers (`LLMOptimizer`, `PPOKernelOptimizer`) for iterative kernel parameter search.
- Ready-made demos, including a Colab notebook and a TinyLlama telemetry pipeline.
- Extensible SDK modules under `agnitra.sdk` for stitching telemetry, optimizers, and deployment runtimes together.

## Installation

Install from source in editable mode along with the recommended extras:

```
pip install -e .[openai,rl]
```

Optional extras unlock additional integrations: install `agnitra[nvml]` for GPU telemetry via NVML, keep `agnitra[openai]` when using the OpenAI Responses API, and add `agnitra[rl]` for Stable Baselines3/Gymnasium support. Export `OPENAI_API_KEY` in your shell or notebook before invoking the LLM-powered helpers.

## Quick Start

### CLI

```
agnitra --help
agnitra-optimize --model demo-model
```

The CLI prints optimization summaries and writes telemetry artifacts (JSON) next to your model inputs. Provide either a model handle or a config file; add `--output` to control the telemetry path.

### SDK

```python
from agnitra.demo import DemoNet

net = DemoNet()
patched_runtime = net.optimize("demo-model")
print(patched_runtime)
```

The SDK mirrors the CLI flow and gives you direct access to optimizers such as `agnitra.core.optimizer.llm_optimizer.LLMOptimizer` and `agnitra.core.optimizer.rl_optimizer.PPOKernelOptimizer` inside your own pipelines.

## Profiling Workflow

Capture telemetry and visualize the resulting FX graph:

1. Profile a model to emit telemetry:
   ```
   python -m cli.main profile tinyllama.pt --input-shape 1,16,64 --output telemetry.json
   ```
2. Load the telemetry and generate a graph IR snapshot:
   ```python
   import json
   import torch

   from agnitra.core.ir.graph_extractor import extract_graph_ir
   from prepare_tinyllama import TinyLlama

   telemetry = json.load(open("telemetry.json"))
   model = TinyLlama().eval()
   example_inputs = (torch.randn(1, 16, 64),)
   graph_ir = extract_graph_ir(model, example_inputs=example_inputs, telemetry=telemetry)
   print(f"Nodes: {len(graph_ir)}")
   print(graph_ir[0])
   ```
3. Open `agnitra_enhanced_demo.ipynb` to explore the graph with ipywidgets, Plotly dashboards, and side-by-side CLI controls.

## Colab Notebook

Launch the hosted demo:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/drvt69talati/agnitraai/blob/main/agnitra_enhanced_demo.ipynb)

In Colab, bootstrap credentials with `google.colab.userdata`:

```python
from google.colab import userdata
import os

userdata.set("OPENAI_API_KEY", "sk-...")
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

The notebook walks through profiling popular open models (LLaMA-3, Whisper, Stable Diffusion) and visualizing telemetry without leaving the browser.

## Project Layout
- `agnitra/` core SDK modules (optimizers, RL policies, telemetry helpers).
- `cli/` command-line entry points (`agnitra`, `agnitra-optimize`).
- `tests/` pytest suite mirroring the package structure.
- `docs/` additional guides including `responses_api.md`.
- `agnitra_enhanced_demo.ipynb` widget-driven sandbox for Graph IR exploration.

## Development

Run the test suite after installing dependencies:

```
pytest -q
```

The harness writes profiling artifacts to the working directory; clear them between runs if you swap models. Document user-visible changes in `README.md` or `docs/` and accompany new behaviour with tests.

## Additional Resources
- `docs/responses_api.md` for the latest OpenAI Responses API specification.
- `docs/non_interactive_codex_usage.txt` for headless Codex integration tips.
- `AGENTS.md` for agent orchestration concepts that back the optimization demos.
- `notes.yaml` for high-level project notes and TODOs.
