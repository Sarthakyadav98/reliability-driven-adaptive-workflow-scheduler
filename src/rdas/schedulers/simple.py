"""Round Robin and Least-Loaded baselines (reliability-blind, with a basic health check)."""

from __future__ import annotations

from ..fault_tolerance import RetryOnly
from .cost import CostModel
from .placement import LeastLoadedPlacement, RoundRobinPlacement
from .policy import PolicyScheduler


class RoundRobin(PolicyScheduler):
    """Cycle through workers in fixed order, skipping workers that are currently down."""

    def __init__(self) -> None:
        super().__init__(RoundRobinPlacement(), RetryOnly(), cost=CostModel(reliability_aware=False),
                         name="round-robin")


class LeastLoaded(PolicyScheduler):
    """Send each ready task to the up worker with the smallest expected backlog."""

    def __init__(self) -> None:
        super().__init__(LeastLoadedPlacement(), RetryOnly(), cost=CostModel(reliability_aware=False),
                         name="least-loaded")
