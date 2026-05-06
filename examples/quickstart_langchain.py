"""Agnitra + LangChain — optimize the LLM behind your agent.

Why this matters: an agent calls the LLM hundreds of times per task
(prompt, tool selection, response, evaluation, retry). A 1.5x speedup
on the model becomes a 1.5x reduction in the agent's wall-clock time.

This example loads a HuggingFacePipeline LLM, optimizes the inner
model with Agnitra, and runs a one-shot prompt — but everything you
do downstream (chains, agents, RAG) inherits the speedup.

Run:
    pip install agnitra langchain langchain-huggingface transformers torchao
    HF_TOKEN=hf_xxx python examples/quickstart_langchain.py
"""
from __future__ import annotations

import os

from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline
import torch

from agnitra.integrations.langchain import optimize_llm

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=os.environ.get("HF_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.float16,
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    pipe = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # The one line: optimize the inner model in place.
    optimize_llm(
        llm,
        agnitra_kwargs={"input_shape": (1, 512), "quantize": "int8_weight"},
    )

    print(llm.invoke("Explain backpropagation in one sentence:"))


if __name__ == "__main__":
    main()
