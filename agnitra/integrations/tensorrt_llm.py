"""TensorRT-LLM backend integration.

This is the move that flips Agnitra from "competing with NVIDIA's stack"
to "driving traffic into NVIDIA's stack." Most HuggingFace developers
cannot use TensorRT-LLM directly — it requires C++, CUDA-version
discipline, and a multi-step engine-build workflow. Agnitra wraps that
complexity behind a single keyword:

    agnitra.optimize(model, backend="tensorrt_llm")

When the wrapper detects an HF model and a TensorRT-LLM install, it
invokes the standard convert-then-build flow under the hood and
returns a runtime that exposes a HuggingFace-shaped ``.generate()``
API. When TensorRT-LLM is missing, the wrapper raises a clear
ImportError pointing at the install docs rather than silently falling
back — TRT-LLM users typically want TRT-LLM specifically.

What this module does NOT do:

* Re-implement TensorRT-LLM kernels. That's NVIDIA's job; we wrap.
* Auto-build the engine from arbitrary architectures. Engine builds
  vary by architecture (Llama vs Mixtral vs Qwen) and are best done
  via TensorRT-LLM's own ``examples/<arch>/convert_checkpoint.py``
  scripts. We accept a pre-built engine path.
* Replace ``agnitra.optimize`` for non-NVIDIA users. The torchao
  path remains the cross-vendor default.

Engine build (one-time, per architecture):

    git clone https://github.com/NVIDIA/TensorRT-LLM
    cd TensorRT-LLM/examples/llama
    python convert_checkpoint.py --model_dir <hf_model> --output_dir ./ckpt --dtype float16
    trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine \\
        --gemm_plugin float16 --max_batch_size 32

Then point Agnitra at the engine:

    result = agnitra.optimize(
        model,
        backend="tensorrt_llm",
        backend_kwargs={"engine_dir": "./engine"},
    )
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

LOGGER = logging.getLogger(__name__)


def _require_tensorrt_llm():
    """Import TensorRT-LLM lazily; surface a clear error when missing."""
    try:
        import tensorrt_llm  # noqa: F401
        from tensorrt_llm.runtime import ModelRunner
    except ImportError as exc:
        raise ImportError(
            "TensorRT-LLM is not installed. Install via the NVIDIA NGC "
            "container or `pip install tensorrt_llm` on a system with a "
            "matching CUDA toolchain. See "
            "https://nvidia.github.io/TensorRT-LLM/ for the official "
            "install path."
        ) from exc
    return ModelRunner


def optimize_with_tensorrt_llm(
    model: Any,
    *,
    engine_dir: str | Path,
    tokenizer: Any = None,
    rank: int = 0,
) -> "TensorRTLLMRuntime":
    """Wrap a pre-built TensorRT-LLM engine in a HuggingFace-shaped runtime.

    ``engine_dir`` must point at a directory produced by ``trtllm-build``.
    The returned object exposes ``.generate(input_ids, ...)`` so existing
    HuggingFace code paths work unchanged. ``model`` is accepted for API
    symmetry with the rest of ``agnitra.optimize`` callers but is not
    used — TensorRT-LLM runs the engine directly.
    """
    ModelRunner = _require_tensorrt_llm()
    engine_path = Path(engine_dir)
    if not engine_path.is_dir():
        raise FileNotFoundError(
            f"engine_dir {engine_dir} does not exist. Build with `trtllm-build` first."
        )
    runner = ModelRunner.from_dir(engine_dir=str(engine_path), rank=rank)
    return TensorRTLLMRuntime(runner=runner, tokenizer=tokenizer)


class TensorRTLLMRuntime:
    """Adapter exposing ``.generate(...)`` over a TensorRT-LLM ModelRunner.

    Mimics ``transformers.PreTrainedModel.generate`` shape closely enough
    for the common path (input_ids, attention_mask, max_new_tokens, do_sample).
    The full ``GenerationMixin`` surface is not implemented; PRs welcome.
    """

    def __init__(self, runner: Any, tokenizer: Any = None) -> None:
        self._runner = runner
        self._tokenizer = tokenizer

    @property
    def device(self):  # noqa: D401 - mimic torch attribute
        try:
            import torch
            return torch.device("cuda")
        except ImportError:  # pragma: no cover
            return None

    def generate(
        self,
        input_ids: Any,
        *,
        max_new_tokens: int = 64,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 1,
        top_p: float = 1.0,
        attention_mask: Any = None,  # noqa: ARG002 - accepted for API symmetry
        **_unused: Any,
    ) -> Any:
        """Run the engine and return generated token IDs.

        ``attention_mask`` is accepted but ignored — TRT-LLM uses
        end_id / pad_id for length control. The shape of the returned
        tensor matches HF: (batch, prompt_len + new_tokens).
        """
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required to use TensorRTLLMRuntime") from exc

        if isinstance(input_ids, torch.Tensor):
            batched = [row.cuda() for row in input_ids]
        else:
            batched = list(input_ids)

        end_id = self._tokenizer.eos_token_id if self._tokenizer is not None else 2
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer is not None and self._tokenizer.pad_token_id is not None
            else end_id
        )

        return self._runner.generate(
            batched,
            max_new_tokens=max_new_tokens,
            end_id=end_id,
            pad_id=pad_id,
            temperature=temperature,
            top_k=top_k if not do_sample else max(top_k, 1),
            top_p=top_p if do_sample else 1.0,
        )


def package_as_nim(
    model_dir: str | Path,
    *,
    output_dir: str | Path,
    target_arch: str = "h100",
    quantize: Optional[str] = "int8_weight",
) -> Path:
    """Package an Agnitra-optimized model as a NIM-compatible container.

    NVIDIA Inference Microservices (NIM) is NVIDIA's preferred packaging
    format for shipping models. This helper produces the on-disk layout
    NIM expects:

        output_dir/
            Dockerfile          # FROM nvcr.io/nim/...
            model_repo/         # Triton-compatible model repository
                <model>/
                    config.pbtxt
                    1/
                        model.py    # Triton Python backend
            agnitra_optimization.json   # records what Agnitra did

    Returns ``output_dir`` once written. The container itself is built
    by the caller (``docker build`` or ``nvcr.io`` push).

    This is intentionally stub-shaped for v0.2.0 — the on-disk layout
    is correct but the per-architecture conversion step requires GPU
    access and per-model work. Treat the output as a starting template.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dockerfile = out / "Dockerfile"
    dockerfile.write_text(_NIM_DOCKERFILE_TEMPLATE.format(
        target_arch=target_arch,
    ))

    model_repo = out / "model_repo" / Path(model_dir).name / "1"
    model_repo.mkdir(parents=True, exist_ok=True)

    config_pbtxt = out / "model_repo" / Path(model_dir).name / "config.pbtxt"
    config_pbtxt.write_text(_TRITON_CONFIG_TEMPLATE.format(
        model_name=Path(model_dir).name,
    ))

    triton_py = model_repo / "model.py"
    triton_py.write_text(_TRITON_PYTHON_BACKEND_TEMPLATE.format(
        model_dir=str(model_dir),
        quantize=quantize or "",
    ))

    manifest = out / "agnitra_optimization.json"
    manifest.write_text(
        '{\n'
        f'  "agnitra_version": "0.2.0",\n'
        f'  "source_model_dir": "{model_dir}",\n'
        f'  "target_arch": "{target_arch}",\n'
        f'  "quantize": "{quantize or "none"}"\n'
        '}\n'
    )

    LOGGER.info("Packaged %s as NIM-compatible container at %s", model_dir, out)
    return out


_NIM_DOCKERFILE_TEMPLATE = """\
# Generated by agnitra package --as-nim
#
# Build:  docker build -t my-org/agnitra-nim:latest .
# Run:    docker run --rm --gpus all -p 8000:8000 my-org/agnitra-nim:latest
#
# This container exposes the model via Triton Inference Server using
# the Python backend that calls Agnitra. For a NIM-blessed deployment
# in production, see https://docs.nvidia.com/nim/ for the official
# microservice template.

FROM nvcr.io/nvidia/tritonserver:24.10-py3

ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir agnitra[quantize] transformers

COPY model_repo /models
COPY agnitra_optimization.json /agnitra_optimization.json

EXPOSE 8000 8001 8002

CMD ["tritonserver", "--model-repository=/models"]
"""


_TRITON_CONFIG_TEMPLATE = """\
name: "{model_name}"
backend: "python"
max_batch_size: 8

input [
  {{
    name: "INPUT_IDS"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

output [
  {{
    name: "OUTPUT_IDS"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }}
]

instance_group [
  {{
    kind: KIND_GPU
    count: 1
  }}
]
"""


_TRITON_PYTHON_BACKEND_TEMPLATE = '''\
"""Triton Python backend that loads an Agnitra-optimized model.

Generated by `agnitra package --as-nim`. Adjust as needed for your
serving SLA — this template runs greedy decode at the configured
max_batch_size. For high-throughput serving, layer vLLM or TRT-LLM
underneath via Agnitra's backend selector.
"""
import json
import numpy as np
import triton_python_backend_utils as pb_utils

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import agnitra


_MODEL_DIR = "{model_dir}"
_QUANTIZE = "{quantize}" or None


class TritonPythonModel:
    def initialize(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(
            _MODEL_DIR, torch_dtype=torch.float16
        ).cuda()
        result = agnitra.optimize(
            model, input_shape=(1, 512), quantize=_QUANTIZE,
        )
        self.model = result.optimized_model

    def execute(self, requests):
        responses = []
        for request in requests:
            in_t = pb_utils.get_input_tensor_by_name(request, "INPUT_IDS")
            input_ids = torch.tensor(in_t.as_numpy(), dtype=torch.long).cuda()
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids=input_ids, max_new_tokens=128, do_sample=False
                )
            out_t = pb_utils.Tensor("OUTPUT_IDS", output_ids.cpu().numpy().astype(np.int64))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out_t]))
        return responses

    def finalize(self):
        pass
'''


__all__ = [
    "optimize_with_tensorrt_llm",
    "TensorRTLLMRuntime",
    "package_as_nim",
]
