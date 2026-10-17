"""Fault model and reproducible fault traces.

Worker crashes follow a Poisson process with a per-worker rate (the exponential
failure model R(t) = exp(-lambda t) common in reliability-aware scheduling).
After a crash the worker is down for an exponentially distributed repair time,
then the process restarts. Optional slowdown episodes stretch task execution,
and degradation events raise a worker's crash rate partway through a run.

A trace depends only on (workers, config, seed), never on scheduling decisions,
so every scheduler compared on the same seed sees exactly the same faults
(common random numbers).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .models import WorkerSpec


@dataclass(frozen=True)
class Degradation:
    """From time ``at`` onward, multiply ``worker``'s crash rate by ``factor``."""

    worker: int
    at: float
    factor: float


@dataclass(frozen=True)
class ScriptedCrash:
    """Deterministic crash for demos and tests (ignored if the worker is already down)."""

    worker: int
    at: float
    downtime: float


@dataclass
class FaultConfig:
    # Probability that an average task on an average worker is hit by a crash.
    # Converted to a fleet-average crash rate: lambda = -ln(1 - p) / ref_duration.
    task_failure_prob: float = 0.1
    mttr: float = 15.0  # mean time to repair after a crash
    slowdown_rate: float = 0.0  # slowdown episodes per unit time per worker
    slowdown_factor: tuple[float, float] = (2.0, 4.0)
    slowdown_duration: float = 40.0  # mean episode length
    degradations: list[Degradation] = field(default_factory=list)
    scripted_crashes: list[ScriptedCrash] = field(default_factory=list)

    def base_rate(self, ref_duration: float) -> float:
        p = self.task_failure_prob
        if p <= 0:
            return 0.0
        if p >= 1:
            raise ValueError("task_failure_prob must be < 1")
        return -math.log(1.0 - p) / ref_duration

    @staticmethod
    def fault_free() -> "FaultConfig":
        return FaultConfig(task_failure_prob=0.0)


class FaultTrace:
    """Lazily generated, seed-deterministic crash and slowdown timeline per worker."""

    def __init__(self, workers: list[WorkerSpec], cfg: FaultConfig, ref_duration: float, seed: int):
        self.workers = workers
        self.cfg = cfg
        self.base_rate = cfg.base_rate(ref_duration)
        self.seed = seed
        n = len(workers)
        self._crash_rng = [np.random.default_rng([seed, w.id, 0]) for w in workers]
        self._down_rng = [np.random.default_rng([seed, w.id, 1]) for w in workers]
        self._slow_rng = [np.random.default_rng([seed, w.id, 2]) for w in workers]
        self._crashes: list[list[tuple[float, float]]] = [[] for _ in range(n)]
        self._up_from = [0.0] * n
        self._slow: list[list[tuple[float, float, float]]] = [[] for _ in range(n)]
        self._slow_cursor = [0.0] * n
        self._scripted = [sorted((c for c in cfg.scripted_crashes if c.worker == w.id), key=lambda c: c.at)
                          for w in workers]
        self._degr = [sorted(((d.at, d.factor) for d in cfg.degradations if d.worker == w.id))
                      for w in workers]

    # ----- crash process -------------------------------------------------
    def rate(self, worker: int, t: float) -> float:
        r = self.base_rate * self.workers[worker].failure_factor
        for at, factor in self._degr[worker]:
            if at <= t:
                r *= factor
        return r

    def _sample_random_crash(self, worker: int, t0: float) -> float:
        """Next event time of a piecewise-constant-rate Poisson process after t0."""
        if self.base_rate <= 0:
            return math.inf
        budget = self._crash_rng[worker].exponential(1.0)
        t = t0
        breaks = [at for at, _ in self._degr[worker] if at > t0] + [math.inf]
        for b in breaks:
            r = self.rate(worker, t)
            if r > 0 and budget <= r * (b - t):
                return t + budget / r
            if r > 0:
                budget -= r * (b - t)
            t = b
        return math.inf

    def _extend(self, worker: int) -> None:
        t = self._up_from[worker]
        rnd = self._sample_random_crash(worker, t)
        scripted = self._scripted[worker]
        while scripted and scripted[0].at < t:
            scripted.pop(0)  # fell inside a downtime window
        if scripted and scripted[0].at <= rnd:
            c = scripted.pop(0)
            crash, down = c.at, c.downtime
        else:
            crash = rnd
            if math.isinf(crash):
                self._crashes[worker].append((math.inf, math.inf))
                return
            down = max(0.1 * self.cfg.mttr, float(self._down_rng[worker].exponential(self.cfg.mttr)))
        self._crashes[worker].append((crash, crash + down))
        self._up_from[worker] = crash + down

    def crash_event(self, worker: int, k: int) -> tuple[float, float]:
        """k-th (crash_time, recover_time) for a worker; (inf, inf) if no more crashes."""
        events = self._crashes[worker]
        while len(events) <= k:
            if events and math.isinf(events[-1][0]):
                return (math.inf, math.inf)
            self._extend(worker)
        return events[k]

    # ----- slowdowns -----------------------------------------------------
    def _windows_until(self, worker: int, horizon: float) -> list[tuple[float, float, float]]:
        cfg = self.cfg
        if cfg.slowdown_rate <= 0:
            return []
        rng = self._slow_rng[worker]
        while self._slow_cursor[worker] <= horizon:
            start = self._slow_cursor[worker] + rng.exponential(1.0 / cfg.slowdown_rate)
            dur = rng.exponential(cfg.slowdown_duration)
            factor = rng.uniform(*cfg.slowdown_factor)
            self._slow[worker].append((start, start + dur, factor))
            self._slow_cursor[worker] = start + dur
        return self._slow[worker]

    def finish_time(self, worker: int, start: float, nominal_duration: float) -> float:
        """Wall-clock finish time of ``nominal_duration`` of work started at ``start``."""
        remaining = nominal_duration
        t = start
        if self.cfg.slowdown_rate <= 0:
            return t + remaining
        windows = self._slow[worker]
        i = 0
        while True:
            # Guarantees every window starting before t + remaining is generated.
            self._windows_until(worker, t + remaining)
            while i < len(windows) and windows[i][1] <= t:
                i += 1
            if i >= len(windows) or windows[i][0] >= t + remaining:
                return t + remaining
            ws, we, f = windows[i]
            if ws > t:
                remaining -= ws - t
                t = ws
            progress = (we - t) / f
            if progress >= remaining:
                return t + remaining * f
            remaining -= progress
            t = we
            i += 1

    def slowdown_windows(self, worker: int, until: float) -> list[tuple[float, float, float]]:
        return [w for w in self._windows_until(worker, until) if w[0] < until]
