"""Agnitra + HuggingFace transformers in 25 lines.

Demonstrates the integration's wedge: change one import + one class
name, and a HuggingFace model is loaded and optimized in one call.

Run:
    pip install agnitra transformers
    HF_TOKEN=hf_xxx python examples/quickstart_hf.py
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoTokenizer

from agnitra.integrations.huggingface import AgnitraModel

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT = "Explain backpropagation in one sentence:"


def main() -> None:
    if not torch.cuda.is_available():
        print("WARNING: running on CPU; latency will not be representative.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AgnitraModel.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
        agnitra_kwargs={"input_shape": (1, 64)},
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"latency: {time.perf_counter() - t0:.3f}s")
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
