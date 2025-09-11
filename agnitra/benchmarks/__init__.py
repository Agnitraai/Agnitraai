"""Benchmarking utilities for Agnitra.

Provides helpers to compare baseline vs optimized model performance and
persist summary metrics for quick inspection.
"""

from .runner import run_benchmark  # re-export

__all__ = ["run_benchmark"]

