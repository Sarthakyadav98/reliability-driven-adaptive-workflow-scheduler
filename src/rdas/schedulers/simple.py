"""Round Robin and Least-Loaded baselines (reliability-blind, with a basic health check)."""

from __future__ import annotations

from .base import Assignment, Decision, Scheduler


class RoundRobin(Scheduler):
    """Cycle through workers in fixed order, skipping workers that are currently down."""

    name = "round-robin"

    def __init__(self) -> None:
        self._next = 0

    def setup(self, view) -> None:
        self._next = 0

    def decide(self, view) -> Decision:
        dec = Decision()
        n = len(view.workers)
        for tid in view.ready_tasks():
            for _ in range(n):
                w = self._next
                self._next = (self._next + 1) % n
                if view.is_up(w):
                    dec.assign.append(Assignment(tid, w))
                    break
        return dec


class LeastLoaded(Scheduler):
    """Send each ready task to the up worker with the smallest expected backlog."""

    name = "least-loaded"

    def decide(self, view) -> Decision:
        dec = Decision()
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        if not up:
            return dec
        load = {w: view.backlog(w) for w in up}
        for tid in view.ready_tasks():
            w = min(up, key=lambda j: (load[j], j))
            dec.assign.append(Assignment(tid, w))
            load[w] += view.nominal_duration(tid, w)
        return dec
