"""Optional dependency helpers for the SDK.

These helpers centralise checks for optional packages used throughout the
project.  They raise informative errors so callers can gracefully handle
missing dependencies and provide guidance to users on how to install them.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def require_openai():
    """Return the ``OpenAI`` client class.

    Raises:
        RuntimeError: If the ``openai`` package is not installed.
    """
    try:  # pragma: no cover - import guarded for optional dependency
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(
            "The openai package is required. Install with 'pip install agnitra[openai]'."
        ) from exc
    return OpenAI


def require_sb3():
    """Return Stable Baselines3 PPO class and Gymnasium module.

    Raises:
        RuntimeError: If the ``stable_baselines3`` or ``gymnasium`` packages are
            not installed.
    """
    try:  # pragma: no cover - import guarded for optional dependency
        from stable_baselines3 import PPO  # type: ignore
        import gymnasium as gym  # type: ignore
    except Exception as exc:  # pragma: no cover - best effort
        raise RuntimeError(
            "RL extras not found. Install with 'pip install agnitra[rl]'."
        ) from exc
    return PPO, gym
