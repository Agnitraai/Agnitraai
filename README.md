# agnitraai


Optional dependencies are grouped as extras and can be installed with pip:

```
pip install agnitra[openai,rl]
```

The ``openai`` extra provides the OpenAI client used for kernel suggestions,
and the ``rl`` extra installs Stable Baselines3 and Gymnasium for the example
reinforcement learning tuner.

## Quick Start

### CLI

Optimize a model from the command line:

```
agnitra-optimize --model demo-model
```

### Graph IR Explorer

Capture telemetry with the CLI and turn it into an FX graph enriched with profiler data.

1. Profile a model to emit telemetry:

   ```
   python -m cli.main profile tinyllama.pt --input-shape 1,16,64 --output telemetry.json --format json
   ```

2. Convert the telemetry into an interactive-friendly IR snapshot:

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

3. Open `agnitra_enhanced_demo.ipynb` for a widget-driven CLI runner and visual dashboards (ipywidgets/Plotly/Matplotlib).

### DemoNet Pipeline

Run the full optimization pipeline programmatically:

```python
from agnitra.demo import DemoNet

net = DemoNet()
patched_runtime = net.optimize("demo-model")
print(patched_runtime)
```

## Colab Demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/agnitraai/agnitraai/blob/main/agnitra_enhanced_demo.ipynb)

In Colab, set your OpenAI credentials using ``google.colab.userdata``:

```python
from google.colab import userdata
import os

# Store the key securely
userdata.set("OPENAI_API_KEY", "sk-...")

# Expose it for the SDK
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

The notebook also demonstrates loading popular open models—LLaMA-3, Whisper,
and Stable Diffusion—and recording telemetry for a single inference pass. A
widget-driven CLI playground plus the IR explorer make it easy to profile,
inspect FX graphs, and visualize hotspots without leaving Colab.

## Requirements

- PyTorch for profiling and telemetry collection.
- Optional extras: `pynvml` for GPU metrics, `transformers` and `diffusers`
  for the sample model profilers.
- A CUDA-capable GPU (e.g. NVIDIA A100) is recommended. In CPU-only
  environments or when optional dependencies are missing the profiling helpers
  will emit warnings and skip heavy models rather than failing hard.

## Testing

Run unit tests locally after installing dependencies:

```
pip install -e .[openai,rl]
pytest
```
