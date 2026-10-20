"""The adaptive scheduler (core contribution) as a configured ``PolicyScheduler``.

    adaptive = Reliability-Aware HEFT placement (or QUBO / exact)
             + expected-cost fault-tolerance policy (replicate only when it pays)
             + reassignment of queued work away from crashed / degraded workers

All three parts read the same online reliability estimate, which closes the loop
monitor -> estimator -> decision -> execution -> monitor.
"""

from __future__ import annotations

from ..fault_tolerance import ExpectedCostFT, RetryOnly
from .cost import CostModel
from .placement import CostPlacement
from .policy import PolicyScheduler


class AdaptiveScheduler(PolicyScheduler):
    def __init__(
        self,
        backend: str = "greedy",
        replication: bool = True,
        reassignment: bool = True,
        reliability_aware: bool = True,
        # Defaults chosen by scripts/tune.py on dev seeds 0-9 (evaluation uses seeds 100+).
        resource_weight: float = 2.0,
        base_resource_weight: float = 0.05,
        risk_weight: float = 1.0,
        gamma: float = 2.0,
        reassign_margin: float = 0.3,
        batch_size: int = 6,
        max_qubo_workers: int = 8,
        qubo_reads: int = 20,
        qubo_sweeps: int = 500,
        penalty_scale: float = 1.0,
        seed: int = 0,
        explain: bool = False,
        name: str | None = None,
    ):
        ft = ExpectedCostFT(resource_weight, base_resource_weight) if replication else RetryOnly()
        placement = CostPlacement(backend, max_qubo_workers, qubo_reads, qubo_sweeps, penalty_scale, seed)
        super().__init__(
            placement, ft, reassignment=reassignment,
            cost=CostModel(reliability_aware, risk_weight, gamma), batch_size=batch_size,
            reassign_margin=reassign_margin, explain=explain,
            name=name or ("adaptive" if backend == "greedy" else f"adaptive-{backend}"))
