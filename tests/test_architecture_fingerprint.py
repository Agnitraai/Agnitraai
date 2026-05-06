"""Tests for architecture-keyed fingerprinting.

The whole point: two models with identical architecture but different
weights (e.g. base Llama-3-8B and a fine-tune) must produce identical
architecture signatures so optimization decisions can be cached and
reused.
"""
from __future__ import annotations

import types

import torch
from torch import nn

from agnitra.core.runtime.fingerprint import (
    architecture_fingerprint,
    architecture_signature,
)


def _llama_like_config(num_layers: int = 32) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model_type="llama",
        hidden_size=4096,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        num_hidden_layers=num_layers,
        vocab_size=128256,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )


def test_arch_fingerprint_uses_config_when_present():
    model = nn.Linear(1, 1)  # body irrelevant when config drives detection
    model.config = _llama_like_config()

    fp = architecture_fingerprint(model)

    assert fp["source"] == "config"
    assert fp["config"]["model_type"] == "llama"
    assert fp["config"]["hidden_size"] == 4096
    assert fp["config"]["num_hidden_layers"] == 32


def test_same_config_different_weights_same_signature():
    """Base model and a fine-tune (different weights, same architecture)
    must produce the same architecture signature."""
    base = nn.Linear(4, 4)
    base.config = _llama_like_config()

    finetune = nn.Linear(4, 4)
    finetune.config = _llama_like_config()
    # Simulate fine-tune: weights diverge.
    with torch.no_grad():
        finetune.weight.add_(0.1)
        finetune.bias.add_(0.5)

    sig_base = architecture_signature(architecture_fingerprint(base))
    sig_finetune = architecture_signature(architecture_fingerprint(finetune))

    assert sig_base == sig_finetune, "fine-tune must reuse base architecture's cache"


def test_different_config_different_signature():
    """Different architectures must hash differently."""
    llama_8b = nn.Linear(1, 1)
    llama_8b.config = _llama_like_config(num_layers=32)

    llama_70b = nn.Linear(1, 1)
    llama_70b.config = _llama_like_config(num_layers=80)

    sig_8b = architecture_signature(architecture_fingerprint(llama_8b))
    sig_70b = architecture_signature(architecture_fingerprint(llama_70b))

    assert sig_8b != sig_70b


def test_different_model_type_different_signature():
    a = nn.Linear(1, 1)
    a.config = _llama_like_config()
    b = nn.Linear(1, 1)
    b.config = _llama_like_config()
    b.config.model_type = "mistral"

    assert architecture_fingerprint(a) != architecture_fingerprint(b)


def test_falls_back_to_structural_when_no_config():
    """Raw torch model without `.config` still produces a stable signature."""

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)

        def forward(self, x):
            return self.norm(self.linear(x))

    fp1 = architecture_fingerprint(_Tiny())
    fp2 = architecture_fingerprint(_Tiny())

    assert fp1["source"] == "structural"
    assert architecture_signature(fp1) == architecture_signature(fp2)


def test_structural_fingerprint_distinguishes_different_topologies():
    class _Shallow(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    class _Deep(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(*[nn.Linear(8, 8) for _ in range(4)])

        def forward(self, x):
            return self.layers(x)

    sig_shallow = architecture_signature(architecture_fingerprint(_Shallow()))
    sig_deep = architecture_signature(architecture_fingerprint(_Deep()))

    assert sig_shallow != sig_deep


def test_signature_is_deterministic_across_runs():
    """Same fingerprint mapping yields same signature on repeated calls."""
    fp = {"source": "config", "config": {"model_type": "llama", "hidden_size": 4096}}
    sigs = {architecture_signature(fp) for _ in range(5)}
    assert len(sigs) == 1
