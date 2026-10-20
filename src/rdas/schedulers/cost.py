"""Shared reliability-aware cost model and per-decision context.

Every component that reasons about "how good is task i on worker j right now"
(reliability-aware placement, the QUBO, the expected-cost fault-tolerance
policy, reassignment) uses this one model, so they never disagree.

    d_ij = work_i / speed_j * perf_j                        (perf = observed slowdown ratio)
    p_ij = P(success) = estimator.p_success(j, d_ij)       (exp(-lambda_hat_j d) for Gamma-Poisson)
    E_ij = avail_j + d_ij + beta * (1 - p_ij) * (d_ij / 2 + r_i)
    w_i  = criticality_i ** gamma

r_i is the mean duration of task i (cost of a retry), and beta is the risk weight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..workflow import criticality, upward_rank


def mean_costs(wf, workers) -> list[float]:
    inv = sum(1.0 / w.speed for w in workers) / len(workers)
    return [t.work * inv for t in wf.tasks]


class CostModel:
    def __init__(self, reliability_aware: bool = True, risk_weight: float = 1.0, gamma: float = 2.0):
        self.reliability_aware = reliability_aware
        self.risk_weight = risk_weight
        self.gamma = gamma

    def setup(self, view) -> None:
        wf = view.workflow
        self.mean_cost = mean_costs(wf, view.workers)
        self.rank = upward_rank(wf, self.mean_cost)
        self.crit = criticality(wf, self.mean_cost)
        self.weight = [max(c, 0.05) ** self.gamma for c in self.crit]

    def p_success(self, view, w: int, d: float) -> float:
        return view.estimator.p_success(w, d) if self.reliability_aware else 1.0

    def dur(self, view, tid: int, w: int) -> float:
        perf = view.estimator.perf_ratio(w) if self.reliability_aware else 1.0
        return view.nominal_duration(tid, w) * perf

    def expected(self, view, tid: int, w: int, avail: float) -> tuple[float, float, float]:
        """(expected completion from now, duration, success probability)."""
        d = self.dur(view, tid, w)
        p = self.p_success(view, w, d)
        e = avail + d + self.risk_weight * (1.0 - p) * (0.5 * d + self.mean_cost[tid])
        return e, d, p

    def avail(self, view, w: int) -> float:
        return view.backlog(w, perf_aware=self.reliability_aware)

    def slack(self, view, tid: int) -> float:
        """Deadline budget left for this task's remaining critical path (inf without a deadline)."""
        dl = view.workflow.deadline
        if dl is None:
            return float("inf")
        return dl - view.now - self.rank[tid]


@dataclass
class DecisionContext:
    """State shared by the fault-tolerance policy and the placement for one ``decide`` call."""

    view: object
    cost: CostModel
    avail: dict  # worker -> expected time until free (updated as copies are placed)
    pressure: float  # waiting + imminent tasks per worker
    stats: dict = field(default_factory=dict)

    def bump(self, key: str, by: float = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + by
