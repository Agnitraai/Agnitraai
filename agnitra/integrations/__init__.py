"""Integrations with the rest of the Python ML ecosystem.

The `agnitra.optimize(model)` SDK is the universal entry point. Modules
in this package wrap that SDK so users of HuggingFace `transformers`
and `accelerate` can opt in by changing a single line of code rather
than learning a new API.

Lazy imports: importing this package does not require `transformers` or
`accelerate` to be installed. The submodules raise a clear
ImportError if the underlying library is missing.
"""
from __future__ import annotations

__all__ = [
    "huggingface",
    "accelerate_helpers",
]
