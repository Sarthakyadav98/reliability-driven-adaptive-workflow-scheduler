"""Interchangeable placement (assignment) backends.

A placement receives a batch of ready tasks, each with the number of copies the
fault-tolerance policy asked for, and returns distinct workers for every copy.
It updates ``ctx.avail`` as it places, so later batches see the new load.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..qubo import AssignmentProblem, QuboSolver, brute_force, greedy


def _unpack(item) -> tuple[int, int, frozenset]:
    """Items are (task, copies) or (task, copies, workers_to_avoid)."""
    tid, k = item[0], item[1]
    return tid, k, (item[2] if len(item) > 2 else frozenset())


def _prefer(pool: list[int], avoid: frozenset, k: int) -> list[int]:
    """Drop avoided workers (retry elsewhere) unless that leaves fewer than k."""
    kept = [w for w in pool if w not in avoid]
    return kept if len(kept) >= k else pool


class Placement(ABC):
    name = "base"

    def setup(self, view) -> None:
        pass

    @abstractmethod
    def place(self, ctx, items: list[tuple]) -> list[list[int]]:
        """``items`` = [(task, copies[, avoid])] in priority order -> list of worker lists."""

    @staticmethod
    def candidates(ctx, k_max: int) -> list[int]:
        view = ctx.view
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        return up if len(up) >= k_max else list(range(len(view.workers)))

    def stats(self) -> dict:
        return {}


class RoundRobinPlacement(Placement):
    """Cycle through up workers in fixed order, ignoring load, speed and reliability."""

    name = "round-robin"

    def __init__(self) -> None:
        self._next = 0

    def place(self, ctx, items):
        view = ctx.view
        n = len(view.workers)
        out = []
        for item in items:
            tid, k, avoid = _unpack(item)
            allowed = set(_prefer([w for w in range(n) if view.is_up(w)], avoid, k))
            chosen: list[int] = []
            for _ in range(n):
                w = self._next
                self._next = (self._next + 1) % n
                if w in allowed and w not in chosen:
                    chosen.append(w)
                    if len(chosen) == k:
                        break
            out.append(chosen)
        return out


class LeastLoadedPlacement(Placement):
    """Up worker(s) with the smallest expected backlog (nominal speeds)."""

    name = "least-loaded"

    def place(self, ctx, items):
        view, load = ctx.view, ctx.avail  # shared across batches of one decision
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        out = []
        for item in items:
            tid, k, avoid = _unpack(item)
            chosen = sorted(_prefer(up, avoid, k), key=lambda j: (load[j], j))[:k]
            for w in chosen:
                load[w] += view.nominal_duration(tid, w)
            out.append(chosen)
        return out


class EFTPlacement(Placement):
    """Reliability-blind earliest finish time (online HEFT processor selection)."""

    name = "eft"

    def place(self, ctx, items):
        view, avail = ctx.view, ctx.avail  # shared across batches of one decision
        up = [w for w in range(len(view.workers)) if view.is_up(w)]
        out = []
        for item in items:
            tid, k, avoid = _unpack(item)
            pool = _prefer(up, avoid, k)
            chosen = sorted(pool, key=lambda w: (avail[w] + view.nominal_duration(tid, w), w))[:k]
            for w in chosen:
                avail[w] += view.nominal_duration(tid, w)
            out.append(chosen)
        return out


class CostPlacement(Placement):
    """Reliability-aware placement on the shared cost model (``schedulers.cost``).

    backend = "greedy"  -> Reliability-Aware HEFT: rank order, cheapest expected completion
    backend = "qubo"    -> decision-scoped QUBO solved by simulated annealing
    backend = "exact"   -> brute-force optimum of the same objective (small batches only)
    """

    def __init__(self, backend: str = "greedy", max_qubo_workers: int = 8, qubo_reads: int = 20,
                 qubo_sweeps: int = 500, penalty_scale: float = 1.0, seed: int = 0):
        if backend not in ("greedy", "qubo", "exact"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.name = {"greedy": "reliability-heft", "qubo": "qubo-sa", "exact": "exact"}[backend]
        self.max_qubo_workers = max_qubo_workers
        self._qubo = QuboSolver(qubo_reads, qubo_sweeps, seed, penalty_scale=penalty_scale) \
            if backend == "qubo" else None
        self.on_problem = None  # optional hook(view, problem) e.g. for the demo
        self._stats = {"n_solver_calls": 0, "solver_time": 0.0, "solver_build_time": 0.0,
                       "solver_anneal_time": 0.0, "solver_vars": 0, "qubo_infeasible": 0,
                       "qubo_infeasible_tasks": 0, "greedy_gap_sum": 0.0}

    def problem(self, ctx, items, workers) -> AssignmentProblem:
        view, cost = ctx.view, ctx.cost
        items = [_unpack(it) for it in items]
        batch = [tid for tid, _, _ in items]
        n, m = len(batch), len(workers)
        scale = view.ref_duration
        c = np.zeros((n, m))
        q = np.zeros((n, n, m))
        dur = np.zeros((n, m))
        for i, tid in enumerate(batch):
            for jj, w in enumerate(workers):
                e, d, _ = cost.expected(view, tid, w, ctx.avail[w])
                c[i, jj] = cost.weight[tid] * e / scale
                dur[i, jj] = d
        # "Retry elsewhere": make workers the task just failed on prohibitively expensive,
        # as long as enough other candidates remain for its copies.
        big = 10.0 * (float(c.max()) + 1.0)
        for i, (_, k, avoid) in enumerate(items):
            cols = [jj for jj, w in enumerate(workers) if w in avoid]
            if cols and m - len(cols) >= k:
                c[i, cols] += big
        for i in range(n):
            for k in range(i + 1, n):
                # Both on worker j: k (lower priority) waits for i's expected duration.
                q[i, k, :] = cost.weight[batch[k]] * dur[i, :] / scale
        return AssignmentProblem(batch, workers, c, q, k=[k for _, k, _ in items])

    def place(self, ctx, items):
        workers = self.candidates(ctx, max(it[1] for it in items))
        if self.backend == "qubo" and len(workers) > self.max_qubo_workers:
            workers = sorted(workers, key=lambda w: ctx.avail[w])[: self.max_qubo_workers]
        prob = self.problem(ctx, items, workers)
        if self.on_problem is not None:
            self.on_problem(ctx.view, prob)
        if self.backend == "greedy" or (len(items) == 1 and items[0][1] == 1):
            cols = greedy(prob)
        elif self.backend == "exact":
            cols = brute_force(prob)
        else:
            cols, info = self._qubo.solve(prob)
            s = self._stats
            s["n_solver_calls"] += 1
            s["solver_time"] += info.seconds
            s["solver_build_time"] += info.build_seconds
            s["solver_anneal_time"] += info.anneal_seconds
            s["solver_vars"] += info.n_vars
            s["qubo_infeasible"] += 0 if info.feasible else 1
            s["qubo_infeasible_tasks"] += info.violations
            s["greedy_gap_sum"] += prob.energy(cols) - prob.energy(greedy(prob))
        sets = AssignmentProblem._sets(cols)
        out = []
        for item, cs in zip(items, sets):
            tid = item[0]
            chosen = [workers[j] for j in cs]
            for w in chosen:
                ctx.avail[w] += ctx.cost.dur(ctx.view, tid, w)
            out.append(chosen)
        return out

    def stats(self) -> dict:
        s = dict(self._stats)
        calls = s["n_solver_calls"]
        s["mean_solver_ms"] = 1000 * s["solver_time"] / calls if calls else 0.0
        s["mean_solver_build_ms"] = 1000 * s["solver_build_time"] / calls if calls else 0.0
        s["mean_solver_anneal_ms"] = 1000 * s["solver_anneal_time"] / calls if calls else 0.0
        s["mean_solver_vars"] = s["solver_vars"] / calls if calls else 0.0
        s["qubo_feasibility_rate"] = 1 - s["qubo_infeasible"] / calls if calls else 1.0
        return s
