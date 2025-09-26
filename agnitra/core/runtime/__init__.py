"""Runtime patching and tuning utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from agnitra.core.kernel import KernelGenerationResult


class RuntimePatcher:
    """Stub runtime patcher that records the path to the injected kernel."""

    def patch(self, kernel: Union[str, KernelGenerationResult]) -> str:
        """Pretend to patch a runtime with the generated kernel.

        The stub stores the basename of the kernel module so integration tests
        can assert that a kernel was emitted without requiring CUDA.
        """

        if isinstance(kernel, KernelGenerationResult):
            descriptor = Path(kernel.module_path).name
        else:
            descriptor = str(kernel)
        return f"Patched<{descriptor}>"

from .tuning import apply_tuning_preset  # re-export

__all__ = ["RuntimePatcher", "apply_tuning_preset"]
