"""Agnitra in 30 lines.

What the wedge promises: drop one line into your inference code and the
model gets faster. This script demonstrates exactly that — load a model,
optimize it, and confirm the result is functionally identical.

For published numbers see benchmarks/llama3_h100/RESULTS.md.

Run:
    pip install agnitra torch transformers
    HF_TOKEN=hf_xxx python examples/quickstart.py
"""
from __future__ import annotations

import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import agnitra

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
PROMPT = "Explain Karpathy's micrograd in one sentence:"


def time_generation(model, tokenizer, prompt: str, max_new_tokens: int = 64) -> tuple[float, str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed, tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: running on CPU. Numbers will not be representative.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    baseline_time, baseline_output = time_generation(model, tokenizer, PROMPT)
    print(f"baseline:  {baseline_time:.3f}s")

    # The one line.
    result = agnitra.optimize(
        model,
        input_shape=(1, 64),
        device=torch.device(device),
        enable_rl=False,
    )
    optimized_time, optimized_output = time_generation(result.optimized_model, tokenizer, PROMPT)
    print(f"optimized: {optimized_time:.3f}s   ({baseline_time / optimized_time:.2f}x)")

    # Sanity: outputs should match (greedy decode is deterministic).
    if baseline_output != optimized_output:
        print("WARNING: outputs differ — optimization changed model behavior.")


if __name__ == "__main__":
    main()
