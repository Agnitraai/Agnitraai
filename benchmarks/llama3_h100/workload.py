"""Frozen workload definition for the Llama-3-8B / H100 benchmark.

The whole point of this directory is reproducibility, so the workload is
defined as data — not parameters passed in at the command line. If you
need a different workload, copy this directory and change the constants
here, don't override at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

INPUT_TOKENS = 512
OUTPUT_TOKENS = 128

BATCH_SIZES: List[int] = [1, 8, 32]

WARMUP_ITERS = 3
MEASURE_ITERS = 10

# Greedy decode — removes sampling-induced variance from comparisons.
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = 1

# Single canonical prompt. We replicate it `batch_size` times to fill a
# batch; padding behavior is each runner's responsibility but the prompt
# itself is identical so prefill cost is comparable.
#
# This prompt was chosen to (a) survive Llama-3 chat template formatting,
# (b) tokenize close to INPUT_TOKENS without truncation, (c) avoid topical
# content that might trigger refusals on aligned variants.
PROMPT = (
    "You are a careful technical reviewer. Below is the abstract of a "
    "research paper. Read it once, then produce a single-paragraph summary "
    "in plain English suitable for a non-specialist reader. Do not add "
    "information that is not in the abstract.\n\n"
    "Abstract: Modern deep learning systems rely on iterative numerical "
    "optimization, in which models are trained by minimizing a scalar loss "
    "function via stochastic gradient descent. While this paradigm has been "
    "remarkably successful, the resulting optimization landscapes are "
    "non-convex and high-dimensional, and their geometric properties remain "
    "incompletely understood. In this work we present a systematic empirical "
    "study of loss landscapes induced by transformer language models at "
    "scales from one hundred million to one hundred billion parameters, "
    "measured along directions sampled both randomly and aligned to "
    "principal components of recent gradient updates. We report three "
    "consistent observations: first, loss curves along random directions "
    "are nearly quadratic across all scales tested; second, loss along "
    "gradient-aligned directions is sharply asymmetric, with much steeper "
    "ascent than descent; and third, the curvature of the asymmetry "
    "decreases monotonically with model scale. We discuss implications for "
    "learning rate schedules, gradient clipping, and the design of "
    "second-order optimizers.\n\n"
    "Summary:"
)


@dataclass(frozen=True)
class Workload:
    model_id: str = MODEL_ID
    input_tokens: int = INPUT_TOKENS
    output_tokens: int = OUTPUT_TOKENS
    batch_sizes: tuple = tuple(BATCH_SIZES)
    warmup_iters: int = WARMUP_ITERS
    measure_iters: int = MEASURE_ITERS
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    top_k: int = TOP_K
    prompt: str = PROMPT

    def prompts(self, batch_size: int) -> List[str]:
        return [self.prompt] * batch_size


WORKLOAD = Workload()
