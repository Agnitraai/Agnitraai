"""Model loader selecting quantized variants based on available memory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - torch is optional for tests
    torch = None  # type: ignore


def available_vram_gb() -> float:
    """Return total available GPU memory in gigabytes.

    Falls back to system RAM when CUDA is unavailable. If detection fails,
    ``0.0`` is returned which triggers selection of the smallest model.
    """

    if torch and torch.cuda.is_available():  # pragma: no branch - simple runtime check
        try:
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
        except Exception:  # pragma: no cover - ignore CUDA errors
            pass
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:  # pragma: no cover - psutil may be missing
        return 0.0


@dataclass
class ModelSpec:
    """Specification for loading a model."""

    model_id: str
    kwargs: Dict[str, Any]
    expected_memory: str


def select_model(quantize: bool, available_gb: float | None = None) -> ModelSpec:
    """Choose a model variant based on available memory.

    Parameters
    ----------
    quantize:
        If ``True``, force 4-bit loading regardless of available memory.
    available_gb:
        Explicit memory amount in GB. When ``None``, memory is detected via
        :func:`available_vram_gb`.
    """

    if available_gb is None:
        available_gb = available_vram_gb()

    if quantize or available_gb < 16:
        return ModelSpec(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            kwargs={"load_in_4bit": True},
            expected_memory="~6GB",
        )

    return ModelSpec(
        model_id="meta-llama/Meta-Llama-3-8B-Instruct",
        kwargs={"torch_dtype": "float16"},
        expected_memory="~16GB",
    )


def load_model(spec: ModelSpec):
    """Load a model using ``transformers`` according to ``spec``.

    Heavy dependencies are imported lazily to keep module importable in
    lightweight environments and during tests.
    """

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs: Dict[str, Any] = {}
    if spec.kwargs.get("load_in_4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
    else:
        kwargs.update(spec.kwargs)

    model = AutoModelForCausalLM.from_pretrained(spec.model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_id)
    return model, tokenizer


def main() -> None:
    """Command line interface for model selection."""

    parser = argparse.ArgumentParser(
        description="Load an LLM suited for the available GPU memory."
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Force 4-bit quantization (~6GB VRAM for LLaMA-3-8B).",
    )
    args = parser.parse_args()

    spec = select_model(args.quantize)
    print(
        f"Selected {spec.model_id} with args {spec.kwargs}. "
        f"Expected VRAM usage {spec.expected_memory}."
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
