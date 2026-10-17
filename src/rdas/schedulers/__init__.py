"""Scheduler registry. Every scheduler runs on the same engine and metrics code."""

from __future__ import annotations

from .adaptive import AdaptiveScheduler
from .base import Assignment, Decision, Scheduler
from .heft import HEFT, EFTRetry, StaticReplication, heft_plan
from .simple import LeastLoaded, RoundRobin

# Factories take keyword options (e.g. seed, explain); baselines ignore them.
REGISTRY = {
    "round-robin": lambda **kw: RoundRobin(),
    "least-loaded": lambda **kw: LeastLoaded(),
    "heft": lambda **kw: HEFT(),
    "eft-retry": lambda **kw: EFTRetry(),
    "replicate-2": lambda **kw: StaticReplication(k=2),
    "adaptive": lambda **kw: AdaptiveScheduler(backend="greedy", **kw),
    "adaptive-qubo": lambda **kw: AdaptiveScheduler(backend="qubo", **kw),
    # Ablations: which part of the adaptive mechanism actually helps?
    "adaptive-no-replication": lambda **kw: AdaptiveScheduler(
        replication=False, name="adaptive-no-replication", **kw),
    "adaptive-no-reassign": lambda **kw: AdaptiveScheduler(
        reassignment=False, name="adaptive-no-reassign", **kw),
}

BASELINES = ["round-robin", "least-loaded", "heft", "eft-retry", "replicate-2"]


def make_scheduler(name: str, **kw) -> Scheduler:
    if name not in REGISTRY:
        raise ValueError(f"unknown scheduler {name!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[name](**kw)


__all__ = [
    "AdaptiveScheduler", "Assignment", "BASELINES", "Decision", "EFTRetry", "HEFT", "LeastLoaded",
    "REGISTRY", "RoundRobin", "Scheduler", "StaticReplication", "heft_plan", "make_scheduler",
]
