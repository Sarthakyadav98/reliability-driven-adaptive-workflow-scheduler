"""Decision-scoped assignment problem and its QUBO formulation.

At one scheduling decision we place a batch of ready tasks i in I onto candidate
workers j in J. The fault-tolerance policy has already fixed how many copies
k_i each task needs (1, or 2 when replicated). Binary variable x_ij = 1 iff a
copy of task i goes to worker j.

    H(x) =  sum_ij c_ij x_ij                          (reliability-aware expected completion cost)
          + sum_{i<k} sum_j q_ikj x_ij x_kj           (queueing interaction: same worker -> one waits)
          + A sum_i (sum_j x_ij - k_i)^2              (exactly k_i copies, on distinct workers)

Copies of one task are automatically on distinct workers (one variable per
pair). Sharing a worker between *different* tasks is allowed, because workers have
queues; it is charged the expected waiting time ``q``, not forbidden. A
"same worker" prohibition would wrongly rule out legitimate sequential tasks.

The classical greedy backend, the QUBO/SA backend and brute force all optimise
this same objective (``AssignmentProblem.energy``), so their solution quality and
solve time can be compared directly (RQ2). Decoded SA samples are checked for
feasibility (exactly k_i copies). Infeasible ones are counted and reported, then
repaired, never silently passed off as valid.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AssignmentProblem:
    tasks: list[int]  # in priority order (earlier = served first on a shared worker)
    workers: list[int]
    c: np.ndarray  # (n, m) linear cost
    q: np.ndarray  # (n, n, m) pairwise cost, only i < k is used
    k: list[int] | None = None  # copies required per task (default 1)
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.k is None:
            self.k = [1] * len(self.tasks)
        self.k = [max(1, min(int(x), len(self.workers))) for x in self.k]

    @property
    def n(self) -> int:
        return len(self.tasks)

    @property
    def m(self) -> int:
        return len(self.workers)

    @staticmethod
    def _sets(assign) -> list[tuple[int, ...]]:
        return [tuple(a) if isinstance(a, (tuple, list)) else (int(a),) for a in assign]

    def energy(self, assign) -> float:
        """Objective of a feasible assignment. ``assign[i]`` is a column index, or a
        tuple of column indices when k_i > 1."""
        sets = self._sets(assign)
        e = float(sum(self.c[i, j] for i in range(self.n) for j in sets[i]))
        for i in range(self.n):
            for k in range(i + 1, self.n):
                for j in set(sets[i]) & set(sets[k]):
                    e += float(self.q[i, k, j])
        return e

    def feasible(self, assign) -> bool:
        sets = self._sets(assign)
        return all(len(set(s)) == self.k[i] and all(0 <= j < self.m for j in s) for i, s in enumerate(sets))

    def penalty_weight(self) -> float:
        """Count penalty A: larger than anything a task could save by dropping a copy."""
        worst = 0.0
        for i in range(self.n):
            pair = sum(float(self.q[min(i, k), max(i, k)].max()) for k in range(self.n) if k != i)
            worst = max(worst, float(self.c[i].max()) + pair)
        return 1.5 * worst + 1.0

    def _pack(self, sets: list[tuple[int, ...]]):
        """Return ints when every task has one copy (the common case), tuples otherwise."""
        if all(x == 1 for x in self.k):
            return [s[0] for s in sets]
        return [tuple(sorted(s)) for s in sets]


def greedy(p: AssignmentProblem):
    """Classical backend: in priority order, give each task its k_i cheapest distinct columns."""
    sets: list[tuple[int, ...]] = []
    for i in range(p.n):
        chosen: list[int] = []
        for _ in range(p.k[i]):
            best_j, best_cost = None, float("inf")
            for j in range(p.m):
                if j in chosen:
                    continue
                cost = p.c[i, j] + sum(p.q[k, i, j] for k in range(i) if j in sets[k])
                if cost < best_cost - 1e-12:
                    best_j, best_cost = j, cost
            chosen.append(best_j)
        sets.append(tuple(chosen))
    return p._pack(sets)


def brute_force(p: AssignmentProblem, limit: int = 500_000):
    """Exact optimum by enumerating every feasible assignment (validation only)."""
    options = [list(itertools.combinations(range(p.m), p.k[i])) for i in range(p.n)]
    size = 1
    for o in options:
        size *= len(o)
    if size > limit:
        raise ValueError(f"brute force too large: {size} assignments")
    best, best_e = None, float("inf")
    for combo in itertools.product(*options):
        e = p.energy(combo)
        if e < best_e - 1e-12:
            best, best_e = list(combo), e
    return p._pack(best)


def to_bqm(p: AssignmentProblem, penalty: float | None = None):
    """Build the dimod BinaryQuadraticModel. Variables are labelled (i, j) by column index."""
    import dimod

    A = penalty if penalty is not None else p.penalty_weight()
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for i in range(p.n):
        k = p.k[i]
        for j in range(p.m):
            # A (sum_j x_ij - k)^2 = A [ (1 - 2k) sum_j x_ij + 2 sum_{j<l} x_ij x_il + k^2 ]  (x^2 = x)
            bqm.add_variable((i, j), float(p.c[i, j]) + A * (1 - 2 * k))
        for j in range(p.m):
            for l in range(j + 1, p.m):
                bqm.add_interaction((i, j), (i, l), 2 * A)
        bqm.offset += A * k * k
    for i in range(p.n):
        for k in range(i + 1, p.n):
            for j in range(p.m):
                if p.q[i, k, j] != 0:
                    bqm.add_interaction((i, j), (k, j), float(p.q[i, k, j]))
    return bqm


def decode(p: AssignmentProblem, sample: dict):
    """Turn a binary sample into an assignment. Returns (assign, n_infeasible_tasks).

    A task is infeasible when the sample gives it a number of copies other than
    k_i. It is repaired with its cheapest columns, and the violation is counted
    and reported.
    """
    sets, violations = [], 0
    for i in range(p.n):
        chosen = [j for j in range(p.m) if sample.get((i, j), 0) == 1]
        if len(chosen) != p.k[i]:
            violations += 1
            if len(chosen) > p.k[i]:
                chosen = sorted(chosen, key=lambda j: p.c[i, j])[: p.k[i]]
            else:
                rest = sorted((j for j in range(p.m) if j not in chosen), key=lambda j: p.c[i, j])
                chosen = chosen + rest[: p.k[i] - len(chosen)]
        sets.append(tuple(chosen))
    return p._pack(sets), violations


@dataclass
class SolveInfo:
    method: str
    seconds: float  # total = build + anneal + decode
    energy: float
    violations: int = 0  # tasks whose decoded copy count was wrong (repaired)
    n_vars: int = 0
    build_seconds: float = 0.0
    anneal_seconds: float = 0.0
    decode_seconds: float = 0.0

    @property
    def feasible(self) -> bool:
        """True if the raw annealer output already satisfied every constraint."""
        return self.violations == 0


class QuboSolver:
    """Simulated-annealing QUBO backend (dwave-samplers). Real D-Wave hardware can be swapped in
    later by passing any dimod sampler (V4 stretch)."""

    def __init__(self, num_reads: int = 20, num_sweeps: int = 500, seed: int = 0, sampler=None,
                 penalty_scale: float = 1.0):
        if sampler is None:
            from dwave.samplers import SimulatedAnnealingSampler

            sampler = SimulatedAnnealingSampler()
        self.sampler = sampler
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.penalty_scale = penalty_scale
        self._rng = np.random.default_rng(seed)

    def solve(self, p: AssignmentProblem):
        t0 = time.perf_counter()
        bqm = to_bqm(p, penalty=self.penalty_scale * p.penalty_weight())
        t1 = time.perf_counter()
        kwargs = {"num_reads": self.num_reads}
        if type(self.sampler).__name__ == "SimulatedAnnealingSampler":
            kwargs.update(num_sweeps=self.num_sweeps, seed=int(self._rng.integers(2**31 - 1)))
        ss = self.sampler.sample(bqm, **kwargs)
        t2 = time.perf_counter()
        assign, viol = decode(p, ss.first.sample)
        t3 = time.perf_counter()
        return assign, SolveInfo("qubo-sa", t3 - t0, p.energy(assign), viol, len(bqm.variables),
                                 build_seconds=t1 - t0, anneal_seconds=t2 - t1, decode_seconds=t3 - t2)
