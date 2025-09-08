"""Telemetry utilities."""

from dataclasses import dataclass, field
from typing import List

@dataclass
class Telemetry:
    """Simple telemetry recorder."""

    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Record a log message."""
        self.logs.append(message)

    def dump(self) -> List[str]:
        """Return collected log messages."""
        return self.logs

__all__ = ["Telemetry"]
