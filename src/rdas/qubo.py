"""Decision-scoped assignment problem and its QUBO formulation.

At one scheduling decision we place a batch of ready tasks i in I onto
candidate workers j in J. Binary variable x_ij = 1 iff task i goes to worker j.

    H(x) =  sum_ij c_ij x_ij                          (expected, reliability-aware completion cost)
          + sum_{i<k} sum_j q_ikj x_ij x_kj           (queueing interaction: same worker -> one waits)
          + A sum_i (sum_j x_ij - 1)^2                (one-hot: each task on exactly one worker)

The classical greedy backend, the QUBO/SA backend and the brute-force exact
solver all optimise this same objective (``AssignmentProblem.energy``), so
their solution quality and solve time can be compared directly (RQ2).
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
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.tasks)

    @property
    def m(self) -> int:
        return len(self.workers)

    def energy(self, assign: list[int] | np.ndarray) -> float:
        """Objective value of a feasible assignment (``assign[i]`` = worker column index)."""
        e = float(sum(self.c[i, assign[i]] for i in range(self.n)))
        for i in range(self.n):
            for k in range(i + 1, self.n):
                if assign[i] == assign[k]:
                    e += float(self.q[i, k, assign[i]])
        return e

    def penalty_weight(self) -> float:
        """One-hot penalty A: larger than anything a task could save by breaking the constraint."""
        worst = 0.0
        for i in range(self.n):
            pair = sum(float(self.q[min(i, k), max(i, k)].max()) for k in range(self.n) if k != i)
            worst = max(worst, float(self.c[i].max()) + pair)
        return 1.5 * worst + 1.0


def greedy(p: AssignmentProblem) -> list[int]:
    """Classical backend: place tasks in priority order on the column with least marginal cost."""
    assign: list[int] = []
    for i in range(p.n):
        best_j, best_cost = 0, float("inf")
        for j in range(p.m):
            cost = p.c[i, j] + sum(p.q[k, i, j] for k in range(i) if assign[k] == j)
            if cost < best_cost - 1e-12:
                best_j, best_cost = j, cost
        assign.append(best_j)
    return assign


def brute_force(p: AssignmentProblem, limit: int = 500_000) -> list[int]:
    """Exact optimum by enumerating all m^n feasible assignments (validation only)."""
    if p.m ** p.n > limit:
        raise ValueError(f"brute force too large: {p.m}^{p.n}")
    best, best_e = None, float("inf")
    for assign in itertools.product(range(p.m), repeat=p.n):
        e = p.energy(assign)
        if e < best_e - 1e-12:
            best, best_e = list(assign), e
    return best


def to_bqm(p: AssignmentProblem, penalty: float | None = None):
    """Build the dimod BinaryQuadraticModel. Variables are labelled (i, j) by column index."""
    import dimod

    A = penalty if penalty is not None else p.penalty_weight()
    bqm = dimod.BinaryQuadraticModel("BINARY")
    for i in range(p.n):
        for j in range(p.m):
            # A (sum_j x_ij - 1)^2 = A [ -sum_j x_ij + 2 sum_{j<l} x_ij x_il + 1 ]  (x^2 = x)
            bqm.add_variable((i, j), float(p.c[i, j]) - A)
        for j in range(p.m):
            for l in range(j + 1, p.m):
                bqm.add_interaction((i, j), (i, l), 2 * A)
        bqm.offset += A
    for i in range(p.n):
        for k in range(i + 1, p.n):
            for j in range(p.m):
                if p.q[i, k, j] != 0:
                    bqm.add_interaction((i, j), (k, j), float(p.q[i, k, j]))
    return bqm


def decode(p: AssignmentProblem, sample: dict) -> tuple[list[int], int]:
    """Turn a binary sample into an assignment; repair one-hot violations. Returns (assign, n_violations)."""
    assign, violations = [], 0
    for i in range(p.n):
        chosen = [j for j in range(p.m) if sample.get((i, j), 0) == 1]
        if len(chosen) != 1:
            violations += 1
            pool = chosen or list(range(p.m))
            chosen = [min(pool, key=lambda j: p.c[i, j])]
        assign.append(chosen[0])
    return assign, violations


@dataclass
class SolveInfo:
    method: str
    seconds: float
    energy: float
    violations: int = 0
    n_vars: int = 0


class QuboSolver:
    """Simulated-annealing QUBO backend (dwave-samplers). Real D-Wave hardware can be swapped in
    later by passing any dimod sampler (V4 stretch)."""

    def __init__(self, num_reads: int = 20, num_sweeps: int = 500, seed: int = 0, sampler=None):
        if sampler is None:
            from dwave.samplers import SimulatedAnnealingSampler

            sampler = SimulatedAnnealingSampler()
        self.sampler = sampler
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self._rng = np.random.default_rng(seed)

    def solve(self, p: AssignmentProblem) -> tuple[list[int], SolveInfo]:
        t0 = time.perf_counter()
        bqm = to_bqm(p)
        kwargs = {"num_reads": self.num_reads}
        if type(self.sampler).__name__ == "SimulatedAnnealingSampler":
            kwargs.update(num_sweeps=self.num_sweeps, seed=int(self._rng.integers(2**31 - 1)))
        ss = self.sampler.sample(bqm, **kwargs)
        assign, viol = decode(p, ss.first.sample)
        dt = time.perf_counter() - t0
        return assign, SolveInfo("qubo-sa", dt, p.energy(assign), viol, len(bqm.variables))
