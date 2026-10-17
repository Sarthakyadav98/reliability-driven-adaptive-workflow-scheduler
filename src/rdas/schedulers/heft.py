"""HEFT (Topcuoglu, Hariri & Wu, IEEE TPDS 2002) plus online EFT-based fault-tolerant baselines."""

from __future__ import annotations

import bisect
from dataclasses import dataclass

from ..models import WorkerSpec, Workflow
from ..workflow import upward_rank
from .base import Assignment, Decision, Scheduler


@dataclass
class HeftPlan:
    proc: list[int]
    start: list[float]
    finish: list[float]
    rank: list[float]

    @property
    def makespan(self) -> float:
        return max(self.finish)


def mean_costs(wf: Workflow, workers: list[WorkerSpec]) -> list[float]:
    inv = sum(1.0 / w.speed for w in workers) / len(workers)
    return [t.work * inv for t in wf.tasks]


def heft_plan(wf: Workflow, workers: list[WorkerSpec]) -> HeftPlan:
    """Static HEFT schedule: decreasing upward rank, insertion-based earliest finish time.

    Communication costs are zero in our model, so rank_u reduces to the
    longest mean-cost path to an exit task.
    """
    n, m = len(wf), len(workers)
    rank = upward_rank(wf, mean_costs(wf, workers))
    topo_pos = {t: i for i, t in enumerate(wf.topo_order)}
    order = sorted(range(n), key=lambda t: (-rank[t], topo_pos[t]))
    slots: list[list[tuple[float, float]]] = [[] for _ in range(m)]  # sorted busy intervals
    proc = [-1] * n
    start = [0.0] * n
    finish = [0.0] * n
    for t in order:
        ready = max((finish[p] for p in wf.tasks[t].preds), default=0.0)
        best = None
        for j in range(m):
            dur = wf.tasks[t].work / workers[j].speed
            est = _earliest_slot(slots[j], ready, dur)
            eft = est + dur
            if best is None or eft < best[0] - 1e-12:
                best = (eft, est, j)
        eft, est, j = best
        bisect.insort(slots[j], (est, eft))
        proc[t], start[t], finish[t] = j, est, eft
    return HeftPlan(proc, start, finish, rank)


def _earliest_slot(busy: list[tuple[float, float]], ready: float, dur: float) -> float:
    """Earliest start >= ready that fits ``dur`` into an idle gap (insertion policy)."""
    t = ready
    for s, e in busy:
        if t + dur <= s + 1e-12:
            return t
        t = max(t, e)
    return t


class HEFT(Scheduler):
    """Static HEFT: the plan is computed once from nominal speeds and never revised.

    At runtime each task is sent to its planned worker as soon as it is ready,
    ordered by planned start time. A failed task is resubmitted to the *same*
    planned worker (HEFT has no notion of faults), so it waits for that worker
    to recover.
    """

    name = "heft"

    def setup(self, view) -> None:
        self.plan = heft_plan(view.workflow, view.workers)

    def decide(self, view) -> Decision:
        dec = Decision()
        for tid in view.ready_tasks():
            dec.assign.append(Assignment(tid, self.plan.proc[tid], priority=self.plan.start[tid]))
        return dec


class _OnlineEFT(Scheduler):
    """Dynamic list scheduling: decreasing rank_u, place on the up worker(s) with earliest finish."""

    copies = 1

    def setup(self, view) -> None:
        self.rank = upward_rank(view.workflow, mean_costs(view.workflow, view.workers))

    def decide(self, view) -> Decision:
        dec = Decision()
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        if not up:
            return dec
        avail = {w: view.backlog(w) for w in up}
        ready = sorted(view.ready_tasks(), key=lambda t: -self.rank[t])
        for tid in ready:
            eft = sorted(up, key=lambda w: (avail[w] + view.nominal_duration(tid, w), w))
            for w in eft[: self.copies]:
                dec.assign.append(Assignment(tid, w, priority=-self.rank[tid]))
                avail[w] += view.nominal_duration(tid, w)
        return dec


class EFTRetry(_OnlineEFT):
    """Online EFT with resubmission on failure to whichever worker now finishes earliest.

    The standard *retry* fault-tolerance policy. It is also the reliability-blind
    twin of the adaptive scheduler (same priorities and placement rule without
    reliability), which makes it the cleanest ablation.
    """

    name = "eft-retry"


class StaticReplication(_OnlineEFT):
    """Static fault-tolerant baseline: every task is replicated on its k earliest-finish workers.

    Uniform k-replication (primary/backup, k = 2) is the classic fixed
    fault-tolerance policy. It applies the same redundancy to every task
    whatever the workers' current reliability, plus retry if all copies fail.
    """

    def __init__(self, k: int = 2):
        self.copies = k
        self.name = f"replicate-{k}"
