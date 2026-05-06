"""Agnitra + LlamaIndex — optimize the LLM behind your RAG / agent flows.

Same compounding pattern as the LangChain example: LlamaIndex hits the
LLM many times (query rewriting, response synthesis, evaluators), so
model-level speedups translate directly into pipeline speedups.

Run:
    pip install agnitra llama-index llama-index-llms-huggingface torchao
    HF_TOKEN=hf_xxx python examples/quickstart_llama_index.py
"""
from __future__ import annotations

import os

from llama_index.llms.huggingface import HuggingFaceLLM
import torch

from agnitra.integrations.llama_index import optimize_llm

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def main() -> None:
    llm = HuggingFaceLLM(
        model_name=MODEL_ID,
        tokenizer_name=MODEL_ID,
        max_new_tokens=64,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16, "token": os.environ.get("HF_TOKEN")},
        tokenizer_kwargs={"token": os.environ.get("HF_TOKEN")},
    )

    # Optimize the underlying model — RAG / agent flows downstream
    # inherit the speedup automatically.
    optimize_llm(
        llm,
        agnitra_kwargs={"input_shape": (1, 512), "quantize": "int8_weight"},
    )

    print(llm.complete("Explain backpropagation in one sentence:").text)


if __name__ == "__main__":
    main()
