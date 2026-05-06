"""Agnitra in 40 lines — runs without an HF token.

Drop-in optimizer for any HuggingFace decoder-LM. This example uses
Phi-3-mini (open weights, ~7 GB) so anyone can run it without a token.
Substitute any other supported architecture (Llama / Mistral / Qwen /
Gemma) and the same pattern works.

Run:
    pip install "agnitra[quantize]" transformers
    python examples/quickstart.py
"""
from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import agnitra

# Open-weight model — no HF_TOKEN required. Swap in any supported
# decoder-LM you have access to: Llama-3, Mistral-7B, Qwen2.5, etc.
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    baseline_time, baseline_output = time_generation(model, tokenizer, PROMPT)
    print(f"baseline:  {baseline_time:.3f}s")

    # The one line. quantize="auto" picks FP8 on Hopper+, INT8 elsewhere.
    result = agnitra.optimize(
        model,
        input_shape=(1, 64),
        device=torch.device(device),
        enable_rl=False,
        quantize="auto" if device == "cuda" else None,
    )
    optimized_time, optimized_output = time_generation(result.optimized_model, tokenizer, PROMPT)
    speedup = baseline_time / optimized_time if optimized_time > 0 else 0.0
    print(f"optimized: {optimized_time:.3f}s   ({speedup:.2f}x)")

    # Sanity: outputs should match (greedy decode is deterministic).
    if baseline_output != optimized_output:
        print("WARNING: outputs differ — optimization changed model behavior.")


if __name__ == "__main__":
    main()
