"""Demonstration script using DemoNet.

This module also provides utilities to load a few popular open models and
record telemetry for them. The additional helpers are intended for use in
GPU backed environments such as Google Colab where the models can actually
be downloaded. Each loader is wrapped in ``try``/``except`` so that missing
dependencies or insufficient resources result in a warning instead of a hard
failure. This keeps the demo script importable in lightweight testing
environments.
"""

from __future__ import annotations

import argparse
import logging

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - exercised when torch absent
    torch = None

from agnitra.sdk import (
    Telemetry,
    IRExtractor,
    LLMOptimizer,
    RLAgent,
    KernelGenerator,
    RuntimePatcher,
)
from agnitra.telemetry_collector import profile_model


class DemoNet:
    """Example network demonstrating SDK usage."""

    def __init__(self) -> None:
        self.telemetry = Telemetry()
        self.extractor = IRExtractor()
        self.optimizer = LLMOptimizer()
        self.agent = RLAgent()
        self.kernel_gen = KernelGenerator()
        self.patcher = RuntimePatcher()

    def optimize(self, model: str = "demo-model") -> str:
        """Run the optimization pipeline and return the patched runtime."""
        self.telemetry.log("Starting optimization")
        ir = self.extractor.extract(model)
        telemetry_payload = {
            "events": [
                {
                    "op": "matmul",
                    "name": "aten::matmul",
                    "shape": [1024, 1024],
                    "cuda_time_ms": 10.2,
                }
            ],
            "notes": "Synthetic telemetry highlighting matmul bottleneck",
        }
        optimized = self.optimizer.optimize(
            ir,
            telemetry=telemetry_payload,
            target_latency_ms=8.0,
        )
        policy = self.agent.learn(optimized)
        kernel = self.kernel_gen.generate(policy)
        result = self.patcher.patch(kernel)
        self.telemetry.log("Optimization complete")
        return result


def _profile_llama3() -> None:
    """Load LLaMA-3 and record telemetry for a single forward pass."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.eval()

    inputs = tokenizer("Hello, world", return_tensors="pt")
    telemetry = profile_model(model, inputs.input_ids)
    print("[TELEMETRY] LLaMA-3", telemetry.get("gpu", {}))


def _profile_whisper() -> None:
    """Load Whisper and record telemetry for a dummy audio sample."""

    from transformers import WhisperModel, WhisperProcessor

    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id)
    model.eval()

    dummy_audio = torch.randn(16000)
    features = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
    telemetry = profile_model(model, features.input_features)
    print("[TELEMETRY] Whisper", telemetry.get("gpu", {}))


def _profile_stable_diffusion() -> None:
    """Load Stable Diffusion and profile the text encoder."""

    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", safety_checker=None
    )
    pipe.text_encoder.eval()

    ids = pipe.tokenizer("a cat", return_tensors="pt").input_ids
    telemetry = profile_model(pipe.text_encoder, ids)
    print("[TELEMETRY] StableDiffusion", telemetry.get("gpu", {}))


def profile_sample_models() -> list[str]:
    """Profile a collection of popular open models.

    Each model is loaded and a tiny dummy inference is executed while
    :func:`agnitra.telemetry_collector.profile_model` captures metrics. Any
    import errors or runtime failures are caught and reported as warnings so the
    function can be executed in minimal environments.

    Returns
    -------
    list[str]
        Names of profiling functions that completed successfully. Failures are
        logged as warnings and omitted from the result list.
    """

    if torch is None:
        logging.warning("torch not available; skipping sample model profiling")
        return []

    successes: list[str] = []
    for fn in (_profile_llama3, _profile_whisper, _profile_stable_diffusion):
        try:
            fn()
            successes.append(fn.__name__)
        except Exception as exc:  # pragma: no cover - heavy deps not in tests
            logging.warning("%s failed: %s", fn.__name__, exc)

    return successes


def main() -> None:
    """Entry point for command line execution.

    The demo can either run the synthetic optimisation pipeline or, when the
    ``--profile-samples`` flag is provided, load a set of public models and
    collect telemetry for them. The latter is primarily meant for interactive
    experimentation in Colab.
    """

    parser = argparse.ArgumentParser(description="Agnitra demo")
    parser.add_argument(
        "--profile-samples",
        action="store_true",
        help="Load LLaMA‑3, Whisper and Stable Diffusion and record telemetry.",
    )
    args = parser.parse_args()

    if args.profile_samples:
        profile_sample_models()
        return

    if torch is not None and torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[WARNING] CUDA not available; running on CPU")
    net = DemoNet()
    patched = net.optimize()
    print(patched)


if __name__ == "__main__":
    main()
