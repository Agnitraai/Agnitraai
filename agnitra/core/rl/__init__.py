"""Reinforcement learning agents."""

class RLAgent:
    """Stub reinforcement learning agent."""

    def learn(self, data: str) -> str:
        """Return a fake policy learned from the data."""
        return f"Policy<{data}>"

from .codex_guided import CodexGuidedAgent  # re-export

__all__ = ["RLAgent", "CodexGuidedAgent"]
