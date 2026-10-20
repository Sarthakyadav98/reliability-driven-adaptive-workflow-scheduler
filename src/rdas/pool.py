"""Heterogeneous worker pool generation."""

from __future__ import annotations

import numpy as np

from .models import WorkerSpec


def make_pool(
    n_workers: int,
    rng: np.random.Generator,
    speed_range: tuple[float, float] = (0.6, 1.6),
    failure_spread: float = 1.0,
    fast_is_flaky: float = 0.7,
) -> list[WorkerSpec]:
    """Generate workers with heterogeneous speed and (hidden) reliability.

    ``failure_spread`` is the sigma of a lognormal over per-worker failure
    factors, normalised so their mean is 1 (the fleet-average crash rate stays
    what the fault config asks for). ``fast_is_flaky`` in [0, 1] controls how
    strongly fast workers tend to be the unreliable ones. That is the
    speed/reliability trade-off that makes speed-only schedulers such as HEFT
    pick risky workers (0 = independent, 1 = perfectly rank-correlated).
    """
    speeds = np.sort(rng.uniform(*speed_range, size=n_workers))
    factors = rng.lognormal(mean=0.0, sigma=failure_spread, size=n_workers)
    factors = factors / factors.mean()
    sorted_factors = np.sort(factors)
    shuffled = rng.permutation(factors)
    # Blend: with probability fast_is_flaky take the rank-correlated factor.
    use_sorted = rng.random(n_workers) < fast_is_flaky
    chosen = np.where(use_sorted, sorted_factors, shuffled)
    chosen = chosen / chosen.mean()
    order = rng.permutation(n_workers)  # worker ids should not reveal speed rank
    pool = []
    for wid, idx in enumerate(order):
        pool.append(WorkerSpec(id=wid, speed=round(float(speeds[idx]), 3),
                               failure_factor=round(float(chosen[idx]), 3)))
    return pool


def demo_pool() -> list[WorkerSpec]:
    """Four hand-picked workers for the review demo, including one fast-but-flaky worker."""
    return [
        WorkerSpec(0, speed=1.0, failure_factor=0.3, label="steady"),
        WorkerSpec(1, speed=1.5, failure_factor=3.0, label="fast-flaky"),
        WorkerSpec(2, speed=0.8, failure_factor=0.2, label="slow-reliable"),
        WorkerSpec(3, speed=1.2, failure_factor=0.5, label="fast"),
    ]
