"""Tests for the TensorRT-LLM integration.

These tests do NOT require tensorrt_llm or torch to be installed.
We monkeypatch the import resolver and the runner factory so the
test verifies API shape and packaging logic without touching real
NVIDIA tooling.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from agnitra.integrations.tensorrt_llm import (
    TensorRTLLMRuntime,
    package_as_nim,
)


# ----- import-failure path ---------------------------------------------------


def test_optimize_with_tensorrt_llm_raises_clear_when_missing(monkeypatch, tmp_path):
    """When tensorrt_llm isn't installed, the wrapper points at install docs."""
    from agnitra.integrations import tensorrt_llm

    def _fail():
        raise ImportError("tensorrt_llm not installed")

    monkeypatch.setattr(tensorrt_llm, "_require_tensorrt_llm", _fail)

    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    with pytest.raises(ImportError) as excinfo:
        tensorrt_llm.optimize_with_tensorrt_llm(model=None, engine_dir=engine_dir)
    assert "tensorrt_llm" in str(excinfo.value).lower()


def test_optimize_with_tensorrt_llm_raises_when_engine_missing(monkeypatch, tmp_path):
    """Even if tensorrt_llm is installed, a missing engine_dir is a clear error."""
    from agnitra.integrations import tensorrt_llm

    monkeypatch.setattr(
        tensorrt_llm, "_require_tensorrt_llm", lambda: object()
    )

    with pytest.raises(FileNotFoundError):
        tensorrt_llm.optimize_with_tensorrt_llm(
            model=None, engine_dir=tmp_path / "does-not-exist"
        )


# ----- TensorRTLLMRuntime adapter --------------------------------------------


def test_runtime_generate_forwards_to_runner():
    """The adapter forwards generate() to the underlying ModelRunner with
    the right HF -> TRT-LLM kwarg translation."""

    captured = {}

    class _FakeRunner:
        def generate(self, batched, *, max_new_tokens, end_id, pad_id, temperature, top_k, top_p):
            captured.update(
                batched=batched,
                max_new_tokens=max_new_tokens,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            return "fake-output"

    class _FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

    rt = TensorRTLLMRuntime(runner=_FakeRunner(), tokenizer=_FakeTokenizer())

    # Use a list-of-list input to skip the torch dependency.
    output = rt.generate(
        [[1, 2, 3]],
        max_new_tokens=64,
        do_sample=False,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
    )
    assert output == "fake-output"
    assert captured["max_new_tokens"] == 64
    assert captured["end_id"] == 2
    assert captured["pad_id"] == 0
    # Greedy mode forces top_p=1.0
    assert captured["top_p"] == 1.0


def test_runtime_falls_back_when_no_tokenizer():
    """No tokenizer means we use sentinel end_id/pad_id."""
    captured = {}

    class _FakeRunner:
        def generate(self, batched, **kwargs):  # noqa: ARG002
            captured.update(kwargs)
            return None

    rt = TensorRTLLMRuntime(runner=_FakeRunner(), tokenizer=None)
    rt.generate([[1, 2]])
    assert captured["end_id"] == 2
    assert captured["pad_id"] == 2


# ----- NIM packaging ---------------------------------------------------------


def test_package_as_nim_writes_expected_layout(tmp_path):
    model_dir = tmp_path / "my-llama"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    output = tmp_path / "nim-pkg"
    result = package_as_nim(
        model_dir=model_dir,
        output_dir=output,
        target_arch="h100",
        quantize="int8_weight",
    )
    assert result == output
    assert (output / "Dockerfile").exists()
    assert (output / "agnitra_optimization.json").exists()
    assert (output / "model_repo" / "my-llama" / "config.pbtxt").exists()
    assert (output / "model_repo" / "my-llama" / "1" / "model.py").exists()


def test_package_as_nim_dockerfile_references_triton_image(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    output = tmp_path / "out"
    package_as_nim(model_dir=model_dir, output_dir=output)
    text = (output / "Dockerfile").read_text()
    assert "nvcr.io/nvidia/tritonserver" in text
    assert "tritonserver" in text  # the CMD invocation


def test_package_as_nim_records_quantize_in_manifest(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    output = tmp_path / "out"
    package_as_nim(model_dir=model_dir, output_dir=output, quantize="int8_weight")
    manifest = (output / "agnitra_optimization.json").read_text()
    assert '"quantize": "int8_weight"' in manifest
    assert '"agnitra_version"' in manifest


def test_package_as_nim_handles_no_quantization(tmp_path):
    model_dir = tmp_path / "m"
    model_dir.mkdir()
    output = tmp_path / "out"
    package_as_nim(model_dir=model_dir, output_dir=output, quantize=None)
    manifest = (output / "agnitra_optimization.json").read_text()
    assert '"quantize": "none"' in manifest
