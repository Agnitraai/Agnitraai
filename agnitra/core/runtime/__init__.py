"""Runtime patching and tuning utilities."""

class RuntimePatcher:
    """Stub runtime patcher."""

    def patch(self, kernel: str) -> str:
        """Pretend to patch a runtime with the generated kernel."""
        return f"Patched<{kernel}>"

from .tuning import apply_tuning_preset  # re-export

__all__ = ["RuntimePatcher", "apply_tuning_preset"]
