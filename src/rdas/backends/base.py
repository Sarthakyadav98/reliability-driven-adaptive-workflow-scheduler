"""Execution backend interface.

The engine owns all logical state (task readiness, attempts, retries, metrics).
A backend only *executes* attempts and reports what happened as normalized
events, so the same engine and scheduler run unchanged in virtual time
(simulation) or against real worker processes and containers (HTTP / Docker).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import Engine
    from ..models import Attempt

# Normalized event kinds shared by every backend.
SUCCEEDED = "succeeded"
FAILED = "failed"  # attempt failed while its worker stayed up; ``category`` says why
WORKER_UNAVAILABLE = "worker_unavailable"  # worker went down; kills its running attempt
WORKER_AVAILABLE = "worker_available"  # worker is back


@dataclass
class Event:
    kind: str
    attempt: int | None = None
    worker: int | None = None
    category: str | None = None
    info: dict = field(default_factory=dict)


class ExecutionBackend(ABC):
    """What the engine needs from an execution environment."""

    engine: "Engine"

    def bind(self, engine: "Engine") -> None:
        """Called once before the run starts."""
        self.engine = engine

    @abstractmethod
    def start(self, attempt: "Attempt") -> None:
        """Begin executing ``attempt`` on ``attempt.worker`` at engine time ``engine.now``."""

    def cancel(self, attempt: "Attempt") -> None:
        """Best-effort stop of a running attempt (a losing replica). Events that still
        arrive for it are ignored by the engine."""

    @abstractmethod
    def next_events(self) -> tuple[float, list[Event]] | None:
        """Block until something happens. Return (engine time, events at that time),
        or None when nothing more can ever happen."""

    def close(self) -> None:
        """Release resources after the run."""
