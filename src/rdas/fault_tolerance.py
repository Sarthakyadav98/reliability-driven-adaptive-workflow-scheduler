"""Fault-tolerance engine: decides *how much protection* a task gets, separately from
*where* it runs.

Every policy returns an explicit ``FTDecision(task, action, replicas, reason)``:
action is "execute" (first attempt, one copy), "retry" (re-run after all copies
failed) or "replicate" (``replicas`` > 1 copies on distinct workers). The
placement backend (round robin, EFT, reliability-aware HEFT, QUBO, ...) then
chooses the workers. Any policy can be paired with any placement.

Retries are bounded by the engine (``max_retries``) and replicas by
``max_replicas``. Nothing is unbounded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class FTDecision:
    task: int
    action: str  # "execute" | "retry" | "replicate"
    replicas: int  # requested number of copies (distinct workers)
    reason: str
    # Workers to avoid if alternatives exist ("reassign": retry somewhere else).
    avoid: frozenset = frozenset()


class FaultTolerancePolicy(ABC):
    name = "base"
    max_replicas = 2
    reassign_on_retry = False  # retry on a *different* worker than the failed one(s)

    def _base_action(self, ctx, tid: int) -> str:
        return "retry" if ctx.view.n_failures(tid) > 0 else "execute"

    def _avoid(self, ctx, tid: int) -> frozenset:
        return frozenset(ctx.view.failed_on(tid)) if self.reassign_on_retry else frozenset()

    @abstractmethod
    def decide(self, ctx, tid: int, avail: dict) -> FTDecision:
        """``avail`` is a provisional worker -> time-until-free map the policy may update."""


class RetryOnly(FaultTolerancePolicy):
    """One copy; after a failure the task is simply retried (the placement re-chooses the worker)."""

    name = "retry"

    def decide(self, ctx, tid, avail):
        action = self._base_action(ctx, tid)
        reason = "single copy" if action == "execute" else "all copies failed: retry"
        return FTDecision(tid, action, 1, reason)


class StaticReplication(FaultTolerancePolicy):
    """Uniform k-replication of every task (classic primary/backup), plus retry."""

    def __init__(self, k: int = 2):
        self.k = k
        self.max_replicas = k
        self.name = f"replicate-{k}"

    def decide(self, ctx, tid, avail):
        return FTDecision(tid, "replicate", self.k, f"static policy: always {self.k} copies")


class ThresholdFT(FaultTolerancePolicy):
    """Rule-based policy from the team's first baseline (Anirudh): replicate when a task
    is critical AND (workers look unreliable OR its deadline slack is low).

    Thresholds are configurable. Slack is measured in mean task durations.
    """

    name = "threshold"
    reassign_on_retry = True

    def __init__(self, criticality_threshold: float = 0.68, reliability_threshold: float = 0.82,
                 slack_threshold: float = 1.0, max_replicas: int = 2):
        self.criticality_threshold = criticality_threshold
        self.reliability_threshold = reliability_threshold
        self.slack_threshold = slack_threshold
        self.max_replicas = max_replicas

    def decide(self, ctx, tid, avail):
        view, cost = ctx.view, ctx.cost
        if view.n_failures(tid) > 0:
            return FTDecision(tid, "retry", 1, "previous attempt failed: retry on another worker",
                              avoid=self._avoid(ctx, tid))
        up = [w for w in range(len(view.workers)) if view.is_up(w)] or list(range(len(view.workers)))
        mean_r = sum(view.estimator.reliability(w, view.ref_duration) for w in up) / len(up)
        slack = cost.slack(view, tid) / view.ref_duration
        critical = cost.crit[tid] >= self.criticality_threshold
        risky = mean_r < self.reliability_threshold
        urgent = slack <= self.slack_threshold
        if critical and (risky or urgent):
            why = "low worker reliability" if risky else "low deadline slack"
            return FTDecision(tid, "replicate", self.max_replicas,
                              f"critical task (C={cost.crit[tid]:.2f}) with {why}")
        return FTDecision(tid, "execute", 1, "standard single-copy execution")


class ExpectedCostFT(FaultTolerancePolicy):
    """Our policy: replicate only when the expected time saved outweighs the extra work.

    With the two best candidate workers a, b (by the shared cost model), copies
    ordered by expected finish f1 <= f2 with success probabilities p1, p2:

        time(single)    = f_a + beta (1 - p_a)(d_a / 2 + r)
        time(replicate) = p1 f1 + (1-p1) p2 f2 + (1-p1)(1-p2)(f2 + beta r)
        res(replicate)  = d1 + E[busy time of the later copy until the earlier wins]
        replicate iff   w_i * (time saved) > rho * (extra busy time)
        rho = base_resource_weight + resource_weight * (waiting + imminent tasks per worker)

    rho is the opportunity cost of busy time: high under backlog, near zero on
    idle capacity (e.g. the tail of a workflow). A replica must be able to start
    before the primary is expected to finish (otherwise it is just a pre-booked
    retry) and never goes to a worker that is down.
    """

    name = "expected-cost"
    reassign_on_retry = True

    def __init__(self, resource_weight: float = 2.0, base_resource_weight: float = 0.05,
                 max_replicas: int = 2):
        self.resource_weight = resource_weight
        self.base_resource_weight = base_resource_weight
        self.max_replicas = max_replicas

    def decide(self, ctx, tid, avail):
        view, cost = ctx.view, ctx.cost
        action = self._base_action(ctx, tid)
        avoid = self._avoid(ctx, tid)
        allowed = [w for w in avail if w not in avoid] or list(avail)
        scored = sorted((cost.expected(view, tid, w, avail[w])[0], w) for w in allowed)
        a = scored[0][1]
        e_a, d_a, p_a = cost.expected(view, tid, a, avail[a])
        f_a = avail[a] + d_a
        avail[a] += d_a  # provisional: later tasks in this batch see the primary's load
        why = "single copy" if action == "execute" else "retry on a different worker"
        if len(allowed) < 2 or self.max_replicas < 2:
            return FTDecision(tid, action, 1, why, avoid=avoid)
        r = cost.mean_cost[tid]
        beta = cost.risk_weight
        rho = self.base_resource_weight + self.resource_weight * ctx.pressure
        single_time = f_a + beta * (1 - p_a) * (0.5 * d_a + r)
        single_res = d_a + (1 - p_a) * r
        best = None
        for b in allowed:
            if b == a or not view.is_up(b) or avail[b] >= f_a:
                continue
            d_b = cost.dur(view, tid, b)
            p_b = cost.p_success(view, b, d_b)
            f_b = avail[b] + d_b
            (f1, d1, p1), (f2, d2, p2) = sorted([(f_a, d_a, p_a), (f_b, d_b, p_b)])
            rep_time = p1 * f1 + (1 - p1) * p2 * f2 + (1 - p1) * (1 - p2) * (f2 + beta * r)
            later_busy = p1 * min(d2, max(0.0, f1 - (f2 - d2))) + (1 - p1) * d2
            rep_res = d1 + later_busy + (1 - p1) * (1 - p2) * r
            gain = cost.weight[tid] * (single_time - rep_time) - rho * (rep_res - single_res)
            if gain > 0 and (best is None or gain > best[0]):
                best = (gain, b, later_busy if f_b >= f_a else d_b, p_b)
        if best is None:
            return FTDecision(tid, action, 1, f"{why} (p_success={p_a:.2f}; replica not worth it)", avoid=avoid)
        gain, b, b_busy, p_b = best
        avail[b] += b_busy
        return FTDecision(tid, "replicate", 2,
                          f"p_success={p_a:.2f} on best worker; replica saves expected time "
                          f"(gain {gain:.2f}, load pressure {ctx.pressure:.1f})", avoid=avoid)


POLICIES = {"retry": RetryOnly, "replicate-2": lambda: StaticReplication(2),
            "threshold": ThresholdFT, "expected-cost": ExpectedCostFT}
