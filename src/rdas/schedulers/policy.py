"""Composed scheduler: fault-tolerance policy + placement backend (+ optional reassignment).

One ``decide`` call:

1. **Reassignment** (optional): queued single-copy tasks whose worker crashed, or
   became clearly worse than an alternative, are withdrawn and moved.
2. Ready tasks, in HEFT upward-rank order, are processed in batches:
   a. the **fault-tolerance policy** decides per task how many copies (execute,
      retry or replicate, with a reason);
   b. the **placement backend** chooses distinct workers for those copies.

Swapping either part changes nothing else, so every combination shares the same
engine, monitor, estimator and metrics.
"""

from __future__ import annotations

from ..fault_tolerance import FaultTolerancePolicy, RetryOnly
from .base import Assignment, Decision, Scheduler
from .cost import CostModel, DecisionContext
from .placement import Placement


class PolicyScheduler(Scheduler):
    def __init__(self, placement: Placement, ft: FaultTolerancePolicy | None = None,
                 reassignment: bool = False, cost: CostModel | None = None, batch_size: int = 6,
                 reassign_margin: float = 0.3, explain: bool = False, name: str | None = None):
        self.placement = placement
        self.ft = ft or RetryOnly()
        self.reassignment = reassignment
        self.cost = cost or CostModel()
        self.batch_size = batch_size
        self.reassign_margin = reassign_margin
        self.explain = explain
        self.name = name or f"{placement.name}+{self.ft.name}"
        self._stats = {"ft_execute": 0, "ft_retry": 0, "ft_replicate": 0, "replicas_requested": 0,
                       "replica_shortfall": 0, "n_reassign_decisions": 0}

    def setup(self, view) -> None:
        self.cost.setup(view)
        self.placement.setup(view)

    def decide(self, view) -> Decision:
        dec = Decision()
        n_w = len(view.workers)
        workers = list(range(n_w))
        avail = {w: self.cost.avail(view, w) for w in workers}
        ready = view.ready_tasks()
        waiting = len(ready) + sum(len(view.queued_tasks(w)) for w in workers) + view.imminent_tasks()
        ctx = DecisionContext(view, self.cost, avail, pressure=waiting / n_w)

        if self.reassignment:
            self._reassign(ctx, dec)

        ready.sort(key=lambda t: -self.cost.rank[t])
        for b in range(0, len(ready), self.batch_size):
            batch = ready[b: b + self.batch_size]
            provisional = dict(ctx.avail)
            decisions = [self.ft.decide(ctx, tid, provisional) for tid in batch]
            placed = self.placement.place(ctx, [(d.task, d.replicas, d.avoid) for d in decisions])
            for d, ws in zip(decisions, placed):
                self._stats[f"ft_{d.action}"] = self._stats.get(f"ft_{d.action}", 0) + 1
                self._stats["replicas_requested"] += d.replicas - 1
                self._stats["replica_shortfall"] += max(0, d.replicas - len(ws))
                for i, w in enumerate(ws):
                    dec.assign.append(Assignment(d.task, w, priority=-self.cost.rank[d.task], reason=d.reason))
                if self.explain and ws:
                    self._explain(ctx, dec, d, ws)
        return dec

    def _explain(self, ctx, dec, d, ws) -> None:
        view, cost = ctx.view, ctx.cost
        tid, a = d.task, ws[0]
        _, _, p_a = cost.expected(view, tid, a, 0.0)
        note = f"{view.workflow.tasks[tid].label} -> W{a} (p_success={p_a:.2f})"
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        if up:
            fastest = min(up, key=lambda w: view.backlog(w) + view.nominal_duration(tid, w))
            if fastest != a:
                note += f"; avoided earliest-finish W{fastest} (p_success={cost.p_success(view, fastest, cost.dur(view, tid, fastest)):.2f})"
        if len(ws) > 1:
            note += f"; REPLICA on W{ws[1]} [{d.reason}]"
        elif d.action == "retry":
            note += " [retry]"
        dec.notes.append(note)

    def _reassign(self, ctx, dec: Decision) -> None:
        view, cost, avail = ctx.view, ctx.cost, ctx.avail
        for w in range(len(view.workers)):
            for tid in view.queued_tasks(w):
                if len(view.copies(tid)) > 1:
                    # Replicated task: just drop a queued copy stuck on a crashed worker.
                    if not view.is_up(w):
                        dec.revoke.append((tid, w))
                        avail[w] -= cost.dur(view, tid, w)
                    continue
                wait = view.wait_before(w, tid, perf_aware=cost.reliability_aware)
                here, d_here, _ = cost.expected(view, tid, w, wait)
                # Never move a task back onto a worker it already failed on: that would undo the
                # retry policy's "retry elsewhere" choice and can bounce the task onto the flaky worker.
                failed = set(view.failed_on(tid))
                alts = [(cost.expected(view, tid, j, avail[j])[0], j) for j in avail
                        if j != w and view.is_up(j) and j not in failed]
                if not alts:
                    continue
                best_alt, j = min(alts)
                if (not view.is_up(w) and best_alt < here) or best_alt + self.reassign_margin * d_here < here:
                    dec.revoke.append((tid, w))
                    avail[w] = max(avail[w] - d_here, 0.0)
                    self._stats["n_reassign_decisions"] += 1
                    why = f"W{w} is down" if not view.is_up(w) else f"W{w} now expected {here:.1f} vs {best_alt:.1f}"
                    if self.explain:
                        dec.notes.append(f"reassign {view.workflow.tasks[tid].label}: {why} -> W{j}")
                    dec.assign.append(Assignment(tid, j, priority=-cost.rank[tid], reason=f"reassigned: {why}"))
                    avail[j] += cost.dur(view, tid, j)

    def stats(self) -> dict:
        s = dict(self._stats)
        s.update(self.placement.stats())
        return s
