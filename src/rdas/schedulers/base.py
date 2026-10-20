"""Scheduler interface: schedulers return decisions, the shared engine executes them."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..engine import SimView


@dataclass
class Assignment:
    """Queue ``task`` on ``worker``. Several assignments of one task = replication."""

    task: int
    worker: int
    priority: float | None = None  # lower runs first in the worker's queue; None = FIFO
    reason: str = ""  # fault-tolerance decision behind this copy (logged per attempt)


@dataclass
class Decision:
    assign: list[Assignment] = field(default_factory=list)
    # (task, worker) pairs: withdraw a *queued* (not yet running) copy, i.e. reassignment.
    revoke: list[tuple[int, int]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)  # human-readable reasons, shown in demo logs


class Scheduler(ABC):
    """Base class. ``decide`` is called after every simulation event.

    It sees the current state through a read-only ``SimView`` and must return
    assignments for (some of) the READY tasks, i.e. tasks whose dependencies are
    done and that have no live copy. Failed tasks come back as READY, so
    retry is just another assignment.
    """

    name: str = "base"

    def setup(self, view: "SimView") -> None:
        """Called once before the first decision (static planners compute their plan here)."""

    @abstractmethod
    def decide(self, view: "SimView") -> Decision:
        ...

    def stats(self) -> dict:
        """Scheduler-specific metrics merged into the run result."""
        return {}
