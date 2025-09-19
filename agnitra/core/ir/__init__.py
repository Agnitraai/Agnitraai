"""IR extraction utilities."""

from __future__ import annotations

from typing import Any, Dict, List


class IRExtractor:
    """Stub intermediate representation extractor with matmul focus."""

    def extract(self, model: str) -> Dict[str, Any]:
        """Return a minimal IR describing a matmul bottleneck for ``model``."""

        return {
            "model": model,
            "nodes": [
                {
                    "name": "matmul_main",
                    "op": "matmul",
                    "shape": [1024, 1024],
                    "input_shapes": [[1024, 1024], [1024, 1024]],
                    "cuda_time_ms": 10.2,
                }
            ],
        }

__all__ = ["IRExtractor"]
