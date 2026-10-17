"""Reliability-aware adaptive scheduler: the core contribution.

Every decision closes the feedback loop

    monitor -> reliability / performance estimates -> placement + fault-handling decision

in three steps:

1. **Reassignment.** Queued copies whose worker has crashed, or has become clearly
   worse than an alternative, are withdrawn and re-placed.
2. **Placement.** Ready tasks (in HEFT upward-rank order) are placed by
   minimising a reliability-aware expected completion cost. This is the decision-
   scoped assignment problem in ``rdas.qubo``, solved by a swappable backend:
   ``greedy`` (classical), ``qubo`` (QUBO + simulated annealing) or ``exact``.
3. **Replication.** For each placed task the policy compares the expected cost
   of a single copy (retry on failure) with that of adding a replica on the
   next-best worker, and replicates only when the criticality-weighted time
   saved outweighs the extra resource use.

Cost model for task i on worker j (``perf`` = observed slowdown ratio):

    d_ij = work_i / speed_j * perf_j
    p_ij = P(success) = exp(-lambda_hat_j * d_ij)          (from the estimator)
    E_ij = avail_j + d_ij + beta * (1 - p_ij) * (d_ij / 2 + r_i)
    (r_i = mean retry duration, beta = risk_weight: >1 also charges the queueing delay a retry suffers)

Task weight w_i = criticality_i ** gamma, so slack-rich tasks matter less.
"""

from __future__ import annotations

import math

import numpy as np

from ..qubo import AssignmentProblem, QuboSolver, brute_force, greedy
from ..workflow import criticality, upward_rank
from .base import Assignment, Decision, Scheduler
from .heft import mean_costs


class AdaptiveScheduler(Scheduler):
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
        seed: int = 0,
        explain: bool = False,
        name: str | None = None,
    ):
        if backend not in ("greedy", "qubo", "exact"):
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.replication = replication
        self.reassignment = reassignment
        self.reliability_aware = reliability_aware
        self.resource_weight = resource_weight
        self.base_resource_weight = base_resource_weight
        self.risk_weight = risk_weight
        self._pressure = 0.0
        self.gamma = gamma
        self.reassign_margin = reassign_margin
        self.batch_size = batch_size
        self.max_qubo_workers = max_qubo_workers
        self.explain = explain
        self.name = name or ("adaptive" if backend == "greedy" else f"adaptive-{backend}")
        self._qubo = QuboSolver(qubo_reads, qubo_sweeps, seed) if backend == "qubo" else None
        self._stats = {"n_replicate_decisions": 0, "n_reassign_decisions": 0,
                       "n_solver_calls": 0, "solver_time": 0.0, "solver_vars": 0,
                       "qubo_violations": 0, "greedy_gap_sum": 0.0}

    # ------------------------------------------------------------------ setup
    def setup(self, view) -> None:
        wf = view.workflow
        self.mean_cost = mean_costs(wf, view.workers)
        self.rank = upward_rank(wf, self.mean_cost)
        crit = criticality(wf, self.mean_cost)
        self.weight = [max(c, 0.05) ** self.gamma for c in crit]

    # ------------------------------------------------------------------ cost model
    def _p_success(self, view, w: int, d: float) -> float:
        return view.estimator.p_success(w, d) if self.reliability_aware else 1.0

    def _dur(self, view, tid: int, w: int) -> float:
        perf = view.estimator.perf_ratio(w) if self.reliability_aware else 1.0
        return view.nominal_duration(tid, w) * perf

    def _expected(self, view, tid: int, w: int, avail: float) -> tuple[float, float, float]:
        """(expected completion from now, duration, success probability)."""
        d = self._dur(view, tid, w)
        p = self._p_success(view, w, d)
        e = avail + d + self.risk_weight * (1.0 - p) * (0.5 * d + self.mean_cost[tid])
        return e, d, p

    def _avail(self, view, w: int, exclude_task: int | None = None) -> float:
        return view.backlog(w, perf_aware=self.reliability_aware, exclude_task=exclude_task)

    # ------------------------------------------------------------------ decide
    def decide(self, view) -> Decision:
        dec = Decision()
        n_w = len(view.workers)
        workers = list(range(n_w))
        to_place = list(view.ready_tasks())
        avail = {w: self._avail(view, w) for w in workers}

        # Load pressure: tasks waiting to start, plus tasks about to become ready, per
        # worker. Replica work only costs something if it delays other work; on idle
        # capacity (e.g. the tail of a workflow) it is nearly free.
        waiting = len(to_place) + sum(len(view.queued_tasks(w)) for w in workers) + view.imminent_tasks()
        self._pressure = waiting / n_w

        if self.reassignment:
            self._reassign(view, dec, avail)

        to_place.sort(key=lambda t: -self.rank[t])
        for b in range(0, len(to_place), self.batch_size):
            batch = to_place[b: b + self.batch_size]
            assign = self._place_batch(view, batch, workers, avail)
            # Primaries first, so replication sees the load of the whole batch.
            placed = [(tid, w, self._place_primary(view, dec, tid, w, avail)) for tid, w in zip(batch, assign)]
            for tid, w, f_a in placed:
                self._maybe_replicate(view, dec, tid, w, f_a, avail)
        return dec

    def _reassign(self, view, dec: Decision, avail: dict) -> None:
        for w in range(len(view.workers)):
            for tid in view.queued_tasks(w):
                copies = view.copies(tid)
                if len(copies) > 1:
                    # Replicated task: just drop a queued copy stuck on a crashed worker.
                    if not view.is_up(w):
                        dec.revoke.append((tid, w))
                        avail[w] -= self._dur(view, tid, w)
                    continue
                wait = view.wait_before(w, tid, perf_aware=self.reliability_aware)
                here, d_here, _ = self._expected(view, tid, w, wait)
                alts = [(self._expected(view, tid, j, avail[j])[0], j) for j in avail if j != w]
                if not alts:
                    continue
                best_alt, j = min(alts)
                if (not view.is_up(w) and best_alt < here) or best_alt + self.reassign_margin * d_here < here:
                    dec.revoke.append((tid, w))
                    avail[w] = max(avail[w] - d_here, 0.0)
                    self._stats["n_reassign_decisions"] += 1
                    if self.explain:
                        dec.notes.append(f"reassign T{tid}: W{w} now expected {here:.1f} vs W{j} {best_alt:.1f}"
                                         + (" (W%d is down)" % w if not view.is_up(w) else ""))
                    # Move it straight to the better worker (no round trip through placement,
                    # which could put it back where it was), never replicating onto w again.
                    f_j = self._place_primary(view, dec, tid, j, avail)
                    self._maybe_replicate(view, dec, tid, j, f_j, avail, exclude=w)

    def _problem(self, view, batch: list[int], workers: list[int], avail: dict) -> AssignmentProblem:
        n, m = len(batch), len(workers)
        scale = view.ref_duration
        c = np.zeros((n, m))
        q = np.zeros((n, n, m))
        dur = np.zeros((n, m))
        for i, tid in enumerate(batch):
            for jj, w in enumerate(workers):
                e, d, _ = self._expected(view, tid, w, avail[w])
                c[i, jj] = self.weight[tid] * e / scale
                dur[i, jj] = d
        for i in range(n):
            for k in range(i + 1, n):
                # Both on worker j: k (lower priority) waits for i's expected duration.
                q[i, k, :] = self.weight[batch[k]] * dur[i, :] / scale
        return AssignmentProblem(batch, workers, c, q)

    def _place_batch(self, view, batch: list[int], workers: list[int], avail: dict) -> list[int]:
        cand = workers
        if self.backend == "qubo" and len(workers) > self.max_qubo_workers:
            # Keep the QUBO small: the workers with the least backlog.
            cand = sorted(workers, key=lambda w: avail[w])[: self.max_qubo_workers]
        prob = self._problem(view, batch, cand, avail)
        if self.backend == "greedy" or len(batch) == 1:
            cols = greedy(prob)
        elif self.backend == "exact":
            cols = brute_force(prob)
        else:
            cols, info = self._qubo.solve(prob)
            g = greedy(prob)
            self._stats["n_solver_calls"] += 1
            self._stats["solver_time"] += info.seconds
            self._stats["solver_vars"] += info.n_vars
            self._stats["qubo_violations"] += info.violations
            self._stats["greedy_gap_sum"] += prob.energy(cols) - prob.energy(g)
        return [cand[j] for j in cols]

    def _place_primary(self, view, dec: Decision, tid: int, a: int, avail: dict) -> float:
        """Queue the primary copy on ``a``; return its expected finish time (relative to now)."""
        e_a, d_a, p_a = self._expected(view, tid, a, avail[a])
        dec.assign.append(Assignment(tid, a, priority=-self.rank[tid]))
        if self.explain:
            fastest = min(avail, key=lambda w: avail[w] + view.nominal_duration(tid, w))
            note = f"T{tid} -> W{a} (p_success={p_a:.2f}, E={e_a:.1f})"
            if fastest != a:
                _, _, p_f = self._expected(view, tid, fastest, avail[fastest])
                note += f"; avoided earliest-finish W{fastest} (p_success={p_f:.2f})"
            dec.notes.append(note)
        f_a = avail[a] + d_a
        avail[a] += d_a
        return f_a

    def _maybe_replicate(self, view, dec: Decision, tid: int, a: int, f_a: float, avail: dict,
                         exclude: int | None = None) -> None:
        """Fault-handling policy: add a replica when its expected cost beats single copy + retry.

        For copies on a and b, ordered by expected finish f1 <= f2 with success
        probabilities p1, p2 and retry duration r:
            time(single)    = f_a + beta (1 - p_a)(d_a / 2 + r)
            time(replicate) = p1 f1 + (1-p1) p2 f2 + (1-p1)(1-p2)(f2 + beta r)
            res(replicate)  = d1 + E[busy time of the later copy until the earlier one wins]
        Replicate iff  w_i * (time saved) > rho * (extra resource), where
            rho = base_resource_weight + resource_weight * (waiting tasks per worker)
        is the opportunity cost of busy time: high under backlog, near zero on idle capacity.
        """
        if not self.replication or len(avail) < 2:
            return
        d_a = self._dur(view, tid, a)
        p_a = self._p_success(view, a, d_a)
        r = self.mean_cost[tid]
        beta = self.risk_weight
        res_w = self.base_resource_weight + self.resource_weight * self._pressure
        single_time = f_a + beta * (1 - p_a) * (0.5 * d_a + r)
        single_res = d_a + (1 - p_a) * r
        best = None
        for b in avail:
            if b == a or b == exclude or not view.is_up(b):
                continue  # a replica is an immediate hedge: never park it on a crashed worker
            d_b = self._dur(view, tid, b)
            if avail[b] >= f_a:
                # Replica would only start after the primary is expected to finish: that is
                # a pre-booked retry, which the retry path already covers, so skip it.
                continue
            p_b = self._p_success(view, b, d_b)
            f_b = avail[b] + d_b
            (f1, d1, p1), (f2, d2, p2) = sorted([(f_a, d_a, p_a), (f_b, d_b, p_b)])
            rep_time = p1 * f1 + (1 - p1) * p2 * f2 + (1 - p1) * (1 - p2) * (f2 + beta * r)
            later_busy = p1 * min(d2, max(0.0, f1 - (f2 - d2))) + (1 - p1) * d2
            rep_res = d1 + later_busy + (1 - p1) * (1 - p2) * r
            gain = self.weight[tid] * (single_time - rep_time) - res_w * (rep_res - single_res)
            if gain > 0 and (best is None or gain > best[0]):
                b_busy = later_busy if f_b >= f_a else d_b
                best = (gain, b, b_busy)
        if best is None:
            return
        _, b, b_busy = best
        dec.assign.append(Assignment(tid, b, priority=-self.rank[tid]))
        avail[b] += b_busy
        self._stats["n_replicate_decisions"] += 1
        if self.explain:
            dec.notes.append(f"  REPLICATE T{tid} on W{b} (p_success on W{a} only {p_a:.2f}; "
                             f"expected time saved outweighs extra work)")

    def stats(self) -> dict:
        s = dict(self._stats)
        calls = s["n_solver_calls"]
        s["mean_solver_ms"] = 1000 * s["solver_time"] / calls if calls else 0.0
        s["mean_solver_vars"] = s["solver_vars"] / calls if calls else 0.0
        return s
