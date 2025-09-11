"""
Prepare a tiny Llama-like model and save it as an eager PyTorch module.

This produces `tinyllama.pt` suitable for the Agnitra profiler. The model is a
compact Transformer block: token embedding + positional encoding, a single
MultiheadAttention layer, and a small MLP. It expects input shaped
`(batch, seq_len, embed_dim)` and returns the same shape.
"""
from __future__ import annotations

from pathlib import Path
import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, S, D)
        s = x.size(1)
        return x + self.pe[:, :s, :]


class TinyLlama(nn.Module):
    def __init__(self, vocab_size: int = 2048, embed_dim: int = 64, n_heads: int = 4, mlp_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, S, D) or token ids (B, S)
        if x.dtype in (torch.long, torch.int64, torch.int32):
            x = self.embed(x)
        x = self.pos(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x


def main() -> int:
    model = TinyLlama()
    # Save TorchScript model for broad compatibility
    scripted = torch.jit.script(model)
    out = Path("tinyllama.pt")
    scripted.save(str(out))

    info = {
        "recommended_input_shape": [1, 16, 64],
        "note": "(batch, seq_len, embed_dim) for floating-point inputs; or pass token ids (1, 16) as integers",
    }
    Path("tinyllama.info.json").write_text(__import__("json").dumps(info, indent=2), encoding="utf-8")

    print(f"Saved tiny model to {out.resolve()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
